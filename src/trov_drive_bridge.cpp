// cmd_vel_rc_bridge.cpp
//
// Nav2 cmd_vel  ->  ArduPilot Rover RC override (native skid-steer mixer).
//
// Subscribes to a Twist (default /cmd_vel_smoothed) and publishes
// mavros_msgs/OverrideRCIn to /mavros/rc/override, driving ONLY the steering
// channel (RC1) and throttle channel (RC3). Every other channel (incl. arm,
// RC6) is published as 0 = "use real RC", so the transmitter keeps arm/mode and
// regains steer+throttle the instant autonomy stops.
//
// NOTE: input defaults to /cmd_vel_smoothed (BEFORE collision_monitor). That
// means this bridge does NOT get collision_monitor's stop gating — chosen
// deliberately so in-place rotation near walls isn't stalled by the stop
// polygon. Obstacle safety then rests on the planner + your remote.
//
// Motion model (matches the hardware):
//   * DRIVE (straight): |w| small. Forward throttle is a FIXED PWM (drive_fwd_thr_pwm,
//             default 1540) whenever v > 0; reverse still scales with v. Small steer,
//             yaw-rate damper (odom) holds heading and kills weave.
//   * ARC   : |w| above w_arc_enter. FIXED PWM pairs per direction:
//               LEFT  (w>0): throttle 1570, steering 1375
//               RIGHT (w<0): throttle 1530, steering 1550
//             Steering is SLEW-LIMITED (arc_steer_slew) so entry is smooth, not a lurch.
//   * STOP  : entered on a FRESH pivot (from driving/idle, or a direction reversal).
//             Snap neutral, hold brake_time_s — this dwell clears the brake-first
//             ESC lockout so a true in-place spin is possible. A pivot resumed in
//             the same direction within pivot_grace_s SKIPS this brake.
//   * PIVOT : in-place rotation, odom closed-loop on yaw RATE (ramps PWM under
//             load). LATCHED: brief cmd_vel dropouts (w flicker) within
//             pivot_grace_s do NOT drop out / re-brake -> continuous rotation.
//             Nav2 still owns heading (when to stop).   [UNCHANGED]
//   * HOLD  : Nav2 commands ~0,~0 (and no pivot latched) -> hold neutral, stay put.
//   * IDLE  : no fresh cmd_vel -> release all channels (remote passes through).
//
// Most numeric params are live-tunable: ros2 param set /cmd_vel_rc_bridge <name> <val>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mavros_msgs/msg/override_rc_in.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

namespace {
constexpr uint16_t RC_RELEASE = 0;   // 0 => "use real RC input" on that channel
constexpr double   EPS         = 1e-6;

double clampd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}
double sgn(double v) { return v >= 0.0 ? 1.0 : -1.0; }
// shortest signed angular difference a-b, wrapped to (-pi, pi]
double ang_diff(double a, double b) { return std::atan2(std::sin(a - b), std::cos(a - b)); }
}  // namespace

class CmdVelRcBridge : public rclcpp::Node {
public:
  CmdVelRcBridge() : Node("cmd_vel_rc_bridge") {
    // ---- topics / timing ----
    input_topic_  = declare_parameter<std::string>("input_topic", "/cmd_vel_smoothed");
    output_topic_ = declare_parameter<std::string>("output_topic", "/mavros/rc/override");
    rate_hz_      = declare_parameter<double>("publish_rate_hz", 25.0);
    watchdog_s_   = declare_parameter<double>("watchdog_s", 0.3);

    // ---- channel map (1-indexed RC channels) ----
    steer_ch_ = declare_parameter<int>("steer_channel", 1);     // RC1
    thr_ch_   = declare_parameter<int>("throttle_channel", 3);  // RC3
    arm_ch_   = declare_parameter<int>("arm_channel", 6);       // RC6 - NEVER overridden

    // ---- PWM shaping ----
    neutral_      = declare_parameter<double>("pwm_neutral", 1501.0);
    thr_span_     = declare_parameter<double>("thr_span_pwm", 80.0);     // thr_n=1 -> +80
    steer_span_   = declare_parameter<double>("steer_span_pwm", 80.0);   // steer_n=1 -> +80
    pivot_pwm_    = declare_parameter<double>("pivot_steer_pwm", 100.0); // pivot feedforward dev
    comfort_cap_  = declare_parameter<double>("comfort_cap", 1.0);       // cap on OUTER wheel = |thr_n|+|steer_n|
    pwm_min_      = declare_parameter<double>("pwm_min", 1200.0);
    pwm_max_      = declare_parameter<double>("pwm_max", 1800.0);

    // ---- cmd_vel -> normalized mapping ----
    v_scale_ = declare_parameter<double>("v_scale", 0.5);  // m/s  -> throttle_n = 1.0
    w_scale_ = declare_parameter<double>("w_scale", 1.0);  // rad/s -> steer_n   = 1.0

    // ---- sign conventions (flip on the bench if it drives/turns wrong) ----
    // steer_sign -1: Nav2 w>0 (CCW/left) -> below-neutral PWM = LEFT on this hw.
    thr_sign_   = declare_parameter<int>("thr_sign", 1)    >= 0 ? 1.0 : -1.0;
    steer_sign_ = declare_parameter<int>("steer_sign", -1) >= 0 ? 1.0 : -1.0;

    // ---- DRIVE: fixed forward throttle PWM ----
    // Used whenever Nav2 commands forward motion (v > 0) in the straight state.
    drive_fwd_thr_pwm_ = declare_parameter<double>("drive_fwd_thr_pwm", 1540.0);

    // ---- ARC: differential (skid) turn, fixed PWM pairs ----
    // When |w| exceeds w_arc_enter the arc state is entered. Absolute PWM values:
    //   LEFT  turn (Nav2 w>0): throttle = arc_left_thr_pwm,  steering = arc_left_steer_pwm
    //   RIGHT turn (Nav2 w<0): throttle = arc_right_thr_pwm, steering = arc_right_steer_pwm
    // Steering is SLEW-LIMITED to arc_steer_slew PWM/s so arc entry ramps instead of lurching.
    // Hysteresis: stays in arc until |w| falls below w_arc_exit.
    w_arc_enter_      = declare_parameter<double>("w_arc_enter", 0.1);         // rad/s
    w_arc_exit_       = declare_parameter<double>("w_arc_exit", 0.08);         // rad/s
    arc_right_thr_pwm_   = declare_parameter<double>("arc_right_thr_pwm", 1530.0);
    arc_right_steer_pwm_ = declare_parameter<double>("arc_right_steer_pwm", 1550.0);
    arc_left_thr_pwm_    = declare_parameter<double>("arc_left_thr_pwm", 1570.0);
    arc_left_steer_pwm_  = declare_parameter<double>("arc_left_steer_pwm", 1375.0);
    arc_steer_slew_   = declare_parameter<double>("arc_steer_slew", 600.0);    // max steer PWM change per second

    // ---- DRIVE (straight): yaw-rate damper (option B) ----
    yaw_damp_enable_ = declare_parameter<bool>("yaw_damp_enable", true);
    yaw_damp_kp_     = declare_parameter<double>("yaw_damp_kp", 0.6);   // rate feedback gain
    yaw_damp_max_    = declare_parameter<double>("yaw_damp_max", 0.4);  // clamp on FB term (rad/s)
    yaw_ema_alpha_   = declare_parameter<double>("yaw_ema_alpha", 0.4); // odom yaw-rate smoothing

    // ---- optional odom forward-speed governor ("don't go too fast") ----
    // 0.0 = OFF. Set e.g. 0.5 once you've confirmed odom-derived speed is clean.
    // NOTE: forward throttle is now a fixed PWM; if the governor trips, forward
    // throttle is scaled down from the fixed deviation toward neutral.
    max_fwd_speed_  = declare_parameter<double>("max_fwd_speed", 0.0);
    spd_ema_alpha_  = declare_parameter<double>("spd_ema_alpha", 0.3);

    // ---- pivot gate (pure-rotation detection) ----
    enable_pivot_ = declare_parameter<bool>("enable_pivot", true);
    v_eps_        = declare_parameter<double>("v_eps", 0.03);          // |v| below = "zero" (HOLD)
    w_eps_        = declare_parameter<double>("w_eps", 0.05);          // |w| below = "zero"
    v_pivot_enter_= declare_parameter<double>("v_pivot_enter", 0.08);  // |v| below this & w present -> pivot
    v_pivot_exit_ = declare_parameter<double>("v_pivot_exit", 0.15);   // |v| above this -> leave pivot (hysteresis)
    brake_time_s_ = declare_parameter<double>("brake_time_s", 1.0);
    // Latch: keep rotating through brief cmd_vel gaps, and skip the re-brake when
    // a same-direction pivot resumes within this window. Must exceed the longest
    // expected w-dropout but should be < a deliberate "change of mind" pause.
    pivot_grace_s_= declare_parameter<double>("pivot_grace_s", 0.5);

    // ---- safety gate ----
    require_armed_ = declare_parameter<bool>("require_armed", true);

    // ---- closed-loop pivot (odom yaw-rate regulation; PIVOT only) ----
    pivot_cl_      = declare_parameter<bool>("pivot_closed_loop", true);
    odom_topic_    = declare_parameter<std::string>("odom_topic", "/odom");
    pivot_window_s_= declare_parameter<double>("pivot_window_s", 0.4);
    pivot_min_dev_ = declare_parameter<double>("pivot_min_dev", 60.0);
    pivot_max_dev_ = declare_parameter<double>("pivot_max_dev", 299.0); // ceiling 1501+299 = 1800
    pivot_step_    = declare_parameter<double>("pivot_step", 20.0);
    pivot_breakaway_step_ = declare_parameter<double>("pivot_breakaway_step", 40.0);
    pivot_rate_tol_= declare_parameter<double>("pivot_rate_tol", 0.05);
    pivot_stuck_rate_ = declare_parameter<double>("pivot_stuck_rate", 0.03);
    odom_timeout_s_= declare_parameter<double>("odom_timeout_s", 0.3);
    over_rate_ratio_ = declare_parameter<double>("pivot_over_rate_ratio", 1.5);
    over_rate_abs_   = declare_parameter<double>("pivot_over_rate_abs", 0.2);
    max_pivot_rate_= declare_parameter<double>("max_pivot_rate", 0.0);

    v_scale_ = std::max(1e-3, v_scale_);
    w_scale_ = std::max(1e-3, w_scale_);
    steer_idx_ = steer_ch_ - 1;
    thr_idx_   = thr_ch_ - 1;
    arm_idx_   = arm_ch_ - 1;

    pub_ = create_publisher<mavros_msgs::msg::OverrideRCIn>(output_topic_, 10);
    sub_ = create_subscription<geometry_msgs::msg::Twist>(
      input_topic_, 10,
      std::bind(&CmdVelRcBridge::on_cmd, this, std::placeholders::_1));
    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", 10,
      std::bind(&CmdVelRcBridge::on_state, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CmdVelRcBridge::on_odom, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CmdVelRcBridge::tick, this));

    param_cb_ = add_on_set_parameters_callback(
      std::bind(&CmdVelRcBridge::on_param_update, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "cmd_vel_rc_bridge up: %s -> %s (steer RC%d, thr RC%d, arm RC%d released) @ %.0f Hz",
      input_topic_.c_str(), output_topic_.c_str(), steer_ch_, thr_ch_, arm_ch_, rate_hz_);
    RCLCPP_WARN(get_logger(),
      "Input is %s (pre-collision_monitor) — no collision gating in the bridge. "
      "Vehicle must be ARMED + MANUAL; SYSID_MYGCS must match MAVROS system_id (1).",
      input_topic_.c_str());
  }

private:
  enum class State { IDLE, DRIVE, ARC, STOP, PIVOT, HOLD };

  // ---------- callbacks ----------
  void on_cmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    last_v_ = msg->linear.x;
    last_w_ = msg->angular.z;
    last_cmd_t_ = now();
    have_cmd_ = true;
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const auto & q = msg->pose.pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    const double px = msg->pose.pose.position.x;
    const double py = msg->pose.pose.position.y;

    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();
    if (have_prev_odom_) {
      const double dt = (t - last_odom_t_).seconds();
      if (dt > 1e-3) {
        const double r = ang_diff(yaw, prev_yaw_) / dt;         // yaw rate, REP-103 (CCW+)
        yaw_rate_meas_ = yaw_ema_alpha_ * r + (1.0 - yaw_ema_alpha_) * yaw_rate_meas_;
        const double dx = px - prev_px_, dy = py - prev_py_;
        const double sp = std::sqrt(dx * dx + dy * dy) / dt;    // speed magnitude (>=0)
        fwd_speed_meas_ = spd_ema_alpha_ * sp + (1.0 - spd_ema_alpha_) * fwd_speed_meas_;
      }
    }
    prev_yaw_ = yaw; prev_px_ = px; prev_py_ = py;
    last_yaw_ = yaw; last_odom_t_ = t;
    have_odom_ = true; have_prev_odom_ = true;
  }

  void on_state(const mavros_msgs::msg::State::SharedPtr msg) {
    if (was_armed_ && !msg->armed) {
      RCLCPP_WARN(get_logger(), "DISARM detected — resetting bridge to IDLE.");
      std::lock_guard<std::mutex> lk(mtx_);
      state_ = State::IDLE;
    }
    was_armed_ = msg->armed;
    is_armed_  = msg->armed;
  }

  rcl_interfaces::msg::SetParametersResult
  on_param_update(const std::vector<rclcpp::Parameter> & params) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto & p : params) {
      const auto & n = p.get_name();
      if      (n == "thr_span_pwm")    thr_span_    = p.as_double();
      else if (n == "steer_span_pwm")  steer_span_  = p.as_double();
      else if (n == "pivot_steer_pwm") pivot_pwm_   = p.as_double();
      else if (n == "comfort_cap")     comfort_cap_ = p.as_double();
      else if (n == "pwm_neutral")     neutral_     = p.as_double();
      else if (n == "v_scale")         v_scale_     = std::max(1e-3, p.as_double());
      else if (n == "w_scale")         w_scale_     = std::max(1e-3, p.as_double());
      else if (n == "thr_sign")        thr_sign_    = p.as_int() >= 0 ? 1.0 : -1.0;
      else if (n == "steer_sign")      steer_sign_  = p.as_int() >= 0 ? 1.0 : -1.0;
      else if (n == "drive_fwd_thr_pwm")    drive_fwd_thr_pwm_    = p.as_double();
      else if (n == "w_arc_enter")     w_arc_enter_  = p.as_double();
      else if (n == "w_arc_exit")      w_arc_exit_   = p.as_double();
      else if (n == "arc_right_thr_pwm")    arc_right_thr_pwm_    = p.as_double();
      else if (n == "arc_right_steer_pwm")  arc_right_steer_pwm_  = p.as_double();
      else if (n == "arc_left_thr_pwm")     arc_left_thr_pwm_     = p.as_double();
      else if (n == "arc_left_steer_pwm")   arc_left_steer_pwm_   = p.as_double();
      else if (n == "arc_steer_slew")  arc_steer_slew_ = p.as_double();
      else if (n == "yaw_damp_enable") yaw_damp_enable_ = p.as_bool();
      else if (n == "yaw_damp_kp")     yaw_damp_kp_  = p.as_double();
      else if (n == "yaw_damp_max")    yaw_damp_max_ = p.as_double();
      else if (n == "yaw_ema_alpha")   yaw_ema_alpha_= clampd(p.as_double(), 0.01, 1.0);
      else if (n == "max_fwd_speed")   max_fwd_speed_= p.as_double();
      else if (n == "spd_ema_alpha")   spd_ema_alpha_= clampd(p.as_double(), 0.01, 1.0);
      else if (n == "enable_pivot")    enable_pivot_ = p.as_bool();
      else if (n == "v_eps")           v_eps_        = p.as_double();
      else if (n == "w_eps")           w_eps_        = p.as_double();
      else if (n == "v_pivot_enter")   v_pivot_enter_= p.as_double();
      else if (n == "v_pivot_exit")    v_pivot_exit_ = p.as_double();
      else if (n == "brake_time_s")    brake_time_s_ = p.as_double();
      else if (n == "pivot_grace_s")   pivot_grace_s_= p.as_double();
      else if (n == "watchdog_s")      watchdog_s_   = p.as_double();
      else if (n == "pivot_closed_loop")     pivot_cl_             = p.as_bool();
      else if (n == "pivot_window_s")        pivot_window_s_       = p.as_double();
      else if (n == "pivot_min_dev")         pivot_min_dev_        = p.as_double();
      else if (n == "pivot_max_dev")         pivot_max_dev_        = p.as_double();
      else if (n == "pivot_step")            pivot_step_           = p.as_double();
      else if (n == "pivot_breakaway_step")  pivot_breakaway_step_ = p.as_double();
      else if (n == "pivot_rate_tol")        pivot_rate_tol_       = p.as_double();
      else if (n == "pivot_stuck_rate")      pivot_stuck_rate_     = p.as_double();
      else if (n == "odom_timeout_s")        odom_timeout_s_       = p.as_double();
      else if (n == "pivot_over_rate_ratio") over_rate_ratio_      = p.as_double();
      else if (n == "pivot_over_rate_abs")   over_rate_abs_        = p.as_double();
      else if (n == "max_pivot_rate")        max_pivot_rate_       = p.as_double();
    }
    rcl_interfaces::msg::SetParametersResult res;
    res.successful = true;
    return res;
  }

  // ---------- publish helpers (single path; arm channel always released) ----------
  mavros_msgs::msg::OverrideRCIn blank() const {
    mavros_msgs::msg::OverrideRCIn mc;
    mc.channels.fill(RC_RELEASE);
    return mc;
  }
  void publish(mavros_msgs::msg::OverrideRCIn mc) {
    if (arm_idx_ >= 0 && static_cast<size_t>(arm_idx_) < mc.channels.size())
      mc.channels[arm_idx_] = RC_RELEASE;
    pub_->publish(mc);
  }
  void publish_release() { publish(blank()); }
  void publish_pair(double steer_pwm, double thr_pwm) {
    auto mc = blank();
    mc.channels[steer_idx_] =
      static_cast<uint16_t>(std::lround(clampd(steer_pwm, pwm_min_, pwm_max_)));
    mc.channels[thr_idx_] =
      static_cast<uint16_t>(std::lround(clampd(thr_pwm, pwm_min_, pwm_max_)));
    publish(mc);
  }
  void publish_neutral() { publish_pair(neutral_, neutral_); }

  // record that we just left STOP/PIVOT, for the skip-rebrake window
  void mark_pivot_exit(const rclcpp::Time & t) {
    last_pivot_exit_t_   = t;
    last_exit_pivot_dir_ = pivot_dir_;
    have_pivot_exit_     = true;
    pivot_win_set_       = false;
  }

  // ---------- main loop ----------
  void tick() {
    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();

    // arm gate
    if (require_armed_ && !is_armed_) {
      if (state_ != State::IDLE)
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
          "DISARMED — releasing all channels to RC.");
      state_ = State::IDLE;
      publish_release();
      return;
    }

    // watchdog: nav silent -> hand everything back to the remote.
    // (keys on message ARRIVAL, not content; pivot latch below handles w flicker)
    const double since = have_cmd_ ? (t - last_cmd_t_).seconds()
                                   : std::numeric_limits<double>::infinity();
    if (since > watchdog_s_) {
      if (state_ != State::IDLE)
        RCLCPP_INFO(get_logger(), "no cmd_vel -> releasing to remote (IDLE)");
      state_ = State::IDLE;
      publish_release();
      return;
    }

    const double v = last_v_, w = last_w_;
    const bool in_pivot_states = (state_ == State::STOP || state_ == State::PIVOT);

    // --- pure-rotation detection (this tick) + latch bookkeeping ---
    const double v_thresh = in_pivot_states ? v_pivot_exit_ : v_pivot_enter_;
    const bool pivot_cmd_now = enable_pivot_
                               && std::fabs(v) < v_thresh
                               && std::fabs(w) > w_eps_;
    if (pivot_cmd_now) {
      last_pivot_cmd_t_  = t;
      last_pivot_dir_    = sgn(steer_sign_ * w);
      last_pivot_target_ = std::fabs(steer_sign_ * w);
      have_pivot_cmd_    = true;
    }
    // Latch keeps us rotating through brief command gaps (only meaningful once
    // we're actually in a pivot state).
    const bool pivot_latched = in_pivot_states && have_pivot_cmd_
                               && (t - last_pivot_cmd_t_).seconds() < pivot_grace_s_;

    // Nav2 commands a stop -> hold position (but NOT while a pivot is latched
    // through a momentary w-dropout).
    if (!pivot_latched && std::fabs(v) < v_eps_ && std::fabs(w) < w_eps_) {
      if (in_pivot_states) mark_pivot_exit(t);
      state_ = State::HOLD;
      publish_neutral();
      log_throttled(v, w);
      return;
    }

    // ================= PIVOT (pure rotation), latched =================
    if (pivot_cmd_now || pivot_latched) {
      if (!in_pivot_states) {
        // fresh entry from driving/idle: brake first UNLESS we were just
        // pivoting the same direction (ESC reverse lockout still clear).
        pivot_dir_     = sgn(steer_sign_ * w);
        pivot_win_set_ = false;
        const bool recent_same_dir =
          have_pivot_exit_
          && (t - last_pivot_exit_t_).seconds() < pivot_grace_s_
          && sgn(pivot_dir_) == sgn(last_exit_pivot_dir_);
        if (recent_same_dir) {
          state_ = State::PIVOT;        // resume immediately, no re-brake
          RCLCPP_INFO(get_logger(), "PIVOT resume (same dir, <%.2fs) — skipping brake",
            pivot_grace_s_);
        } else {
          state_ = State::STOP;
          stop_start_ = t;
        }
      } else if (pivot_cmd_now && sgn(steer_sign_ * w) != sgn(pivot_dir_)) {
        // genuine direction reversal: must re-brake to clear lockout the other way
        pivot_dir_     = sgn(steer_sign_ * w);
        pivot_win_set_ = false;
        state_ = State::STOP;
        stop_start_ = t;
        RCLCPP_INFO(get_logger(), "PIVOT direction reversed — re-braking");
      }

      if (state_ == State::STOP) {
        publish_neutral();                 // full stop clears ESC lockout
        if ((t - stop_start_).seconds() >= brake_time_s_)
          state_ = State::PIVOT;
        log_throttled(v, w);
        return;
      }

      // PIVOT: regulate yaw rate to the last commanded target (held through gaps)
      double target = last_pivot_target_;
      if (max_pivot_rate_ > EPS) target = std::min(target, max_pivot_rate_);
      const double dev = pivot_closed_loop_dev(target, t);
      publish_pair(neutral_ + pivot_dir_ * dev, neutral_);
      log_throttled(v, w);
      return;
    }

    // not pivoting this tick: if we were, record the exit for the skip window
    if (in_pivot_states) mark_pivot_exit(t);

    // --- ARC vs STRAIGHT (hysteresis on |w|) ---
    const double w_arc_thresh = (state_ == State::ARC) ? w_arc_exit_ : w_arc_enter_;

    if (std::fabs(w) > w_arc_thresh) {
      // ============ ARC: fixed PWM pair per direction + slew-limited steering ============
      const bool fresh_arc = (state_ != State::ARC);
      state_ = State::ARC;

      // Nav2 w>0 = LEFT (CCW), w<0 = RIGHT (CW)
      const bool   turn_left = (w > 0.0);
      const double steer_pwm_cmd = turn_left ? arc_left_steer_pwm_  : arc_right_steer_pwm_;
      const double thr_pwm_cmd   = turn_left ? arc_left_thr_pwm_    : arc_right_thr_pwm_;

      // steering deviation from neutral that we want to reach
      double steer_target = steer_pwm_cmd - neutral_;

      // slew-rate limit so arc entry ramps smoothly instead of lurching
      if (fresh_arc) { arc_steer_out_ = 0.0; arc_prev_set_ = false; }  // start ramp from neutral
      if (arc_steer_slew_ > EPS) {
        const double dt = arc_prev_set_
          ? std::max(1e-3, (t - arc_prev_t_).seconds()) : (1.0 / std::max(1.0, rate_hz_));
        const double max_step = arc_steer_slew_ * dt;
        steer_target = clampd(steer_target, arc_steer_out_ - max_step, arc_steer_out_ + max_step);
      }
      arc_steer_out_ = steer_target;
      arc_prev_t_    = t;
      arc_prev_set_  = true;

      publish_pair(neutral_ + steer_target, thr_pwm_cmd);
      log_throttled(v, w);
      return;
    }

    // ============ STRAIGHT: throttle on + small steer + yaw-rate damper ============
    state_ = State::DRIVE;
    double thr_n = clampd(thr_sign_ * v / v_scale_, -1.0, 1.0);

    // optional odom forward-speed governor
    if (max_fwd_speed_ > EPS && have_odom_ && thr_n > 0.0
        && fwd_speed_meas_ > max_fwd_speed_) {
      thr_n *= clampd(max_fwd_speed_ / std::max(fwd_speed_meas_, 1e-3), 0.0, 1.0);
    }

    // yaw-rate damper: setpoint is ALWAYS Nav2's w (~0 here) -> holds heading straight
    double eff_rate = w;
    if (yaw_damp_enable_ && have_odom_) {
      double fb = yaw_damp_kp_ * (w - yaw_rate_meas_);
      fb = clampd(fb, -yaw_damp_max_, yaw_damp_max_);
      eff_rate += fb;
    }
    double steer_n = clampd(steer_sign_ * eff_rate / w_scale_, -1.0, 1.0);

    const double s = std::fabs(thr_n) + std::fabs(steer_n);
    if (s > comfort_cap_ && s > EPS) {
      const double k = comfort_cap_ / s; thr_n *= k; steer_n *= k;
    }

    // Throttle PWM: forward = FIXED drive_fwd_thr_pwm (default 1540);
    // reverse keeps the original proportional mapping.
    double thr_pwm;
    if (thr_n > 0.0) {
      double fwd_dev = drive_fwd_thr_pwm_ - neutral_;
      // keep the speed governor effective on the fixed forward throttle
      if (max_fwd_speed_ > EPS && have_odom_ && fwd_speed_meas_ > max_fwd_speed_) {
        fwd_dev *= clampd(max_fwd_speed_ / std::max(fwd_speed_meas_, 1e-3), 0.0, 1.0);
      }
      thr_pwm = neutral_ + fwd_dev;
    } else {
      thr_pwm = neutral_ + thr_n * thr_span_;
    }

    publish_pair(neutral_ + steer_n * steer_span_, thr_pwm);
    log_throttled(v, w);
  }

  // Window-integrated yaw-rate regulator for in-place rotation. Returns the
  // steering PWM *deviation* (>=0; direction applied by caller).
  double pivot_closed_loop_dev(double target_rate, const rclcpp::Time & t) {
    if (!pivot_cl_)
      return clampd(pivot_pwm_, pivot_min_dev_, pivot_max_dev_);

    if (!pivot_win_set_) {
      pivot_dev_       = pivot_pwm_;      // feedforward seed
      pivot_win_start_ = t;
      pivot_win_yaw_   = last_yaw_;
      pivot_win_set_   = true;
      return clampd(pivot_dev_, pivot_min_dev_, pivot_max_dev_);
    }

    const double odom_age = have_odom_
      ? (t - last_odom_t_).seconds() : std::numeric_limits<double>::infinity();
    if (odom_age > odom_timeout_s_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "PIVOT: odom stale (%.2fs) -> open-loop fallback (dev=%.0f)", odom_age, pivot_pwm_);
      return clampd(pivot_pwm_, pivot_min_dev_, pivot_max_dev_);
    }

    const double win_dt = (t - pivot_win_start_).seconds();
    if (win_dt >= pivot_window_s_) {
      const double measured =
        std::fabs(ang_diff(last_yaw_, pivot_win_yaw_)) / std::max(1e-3, win_dt);

      if (measured > target_rate * over_rate_ratio_ && measured > over_rate_abs_) {
        pivot_dev_ = std::max(pivot_dev_ - 2.0 * pivot_step_, pivot_min_dev_);
      } else if (measured < pivot_stuck_rate_) {
        pivot_dev_ = std::min(pivot_dev_ + pivot_breakaway_step_, pivot_max_dev_);
      } else if (target_rate - measured > pivot_rate_tol_) {
        pivot_dev_ = std::min(pivot_dev_ + pivot_step_, pivot_max_dev_);
      } else if (measured - target_rate > pivot_rate_tol_) {
        pivot_dev_ = std::max(pivot_dev_ - pivot_step_, pivot_min_dev_);
      }

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
        "PIVOT loop: target=%.2f meas=%.2f rad/s -> dev=%.0f (PWM %.0f)  [effort=load]",
        target_rate, measured, pivot_dev_, neutral_ + pivot_dev_);

      pivot_win_start_ = t;
      pivot_win_yaw_   = last_yaw_;
    }
    return clampd(pivot_dev_, pivot_min_dev_, pivot_max_dev_);
  }

  void log_throttled(double v, double w) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[%s] v=%+.2f w=%+.2f  yaw_rate=%+.2f spd=%.2f",
      state_name(), v, w, yaw_rate_meas_, fwd_speed_meas_);
  }
  const char * state_name() const {
    switch (state_) {
      case State::IDLE:  return "IDLE";
      case State::DRIVE: return "DRIVE";
      case State::ARC:   return "ARC";
      case State::STOP:  return "STOP";
      case State::PIVOT: return "PIVOT";
      case State::HOLD:  return "HOLD";
    }
    return "?";
  }

  // ---------- members ----------
  std::string input_topic_, output_topic_;
  double rate_hz_{25.0}, watchdog_s_{0.3};
  int steer_ch_{1}, thr_ch_{3}, arm_ch_{6};
  int steer_idx_{0}, thr_idx_{2}, arm_idx_{5};

  double neutral_{1501.0}, thr_span_{80.0}, steer_span_{80.0}, pivot_pwm_{100.0};
  double comfort_cap_{1.0}, pwm_min_{1000.0}, pwm_max_{2000.0};
  double v_scale_{0.5}, w_scale_{1.0}, thr_sign_{1.0}, steer_sign_{-1.0};

  double drive_fwd_thr_pwm_{1540.0};

  double w_arc_enter_{0.1}, w_arc_exit_{0.08};
  double arc_right_thr_pwm_{1530.0}, arc_right_steer_pwm_{1550.0};
  double arc_left_thr_pwm_{1570.0},  arc_left_steer_pwm_{1375.0};
  double arc_steer_slew_{600.0};
  bool   yaw_damp_enable_{true};
  double yaw_damp_kp_{0.6}, yaw_damp_max_{0.4}, yaw_ema_alpha_{0.4};
  double max_fwd_speed_{0.0}, spd_ema_alpha_{0.3};

  bool   enable_pivot_{true};
  double v_eps_{0.03}, w_eps_{0.05}, v_pivot_enter_{0.08}, v_pivot_exit_{0.15}, brake_time_s_{1.0};
  double pivot_grace_s_{0.5};
  bool   require_armed_{true};

  bool   pivot_cl_{true};
  std::string odom_topic_{"/odom"};
  double pivot_window_s_{0.4}, pivot_min_dev_{60.0}, pivot_max_dev_{299.0};
  double pivot_step_{20.0}, pivot_breakaway_step_{40.0};
  double pivot_rate_tol_{0.05}, pivot_stuck_rate_{0.03}, odom_timeout_s_{0.3};
  double over_rate_ratio_{1.5}, over_rate_abs_{0.2}, max_pivot_rate_{0.0};

  std::mutex mtx_;
  double last_v_{0.0}, last_w_{0.0};
  rclcpp::Time last_cmd_t_;
  bool have_cmd_{false};
  bool is_armed_{false}, was_armed_{false};

  State state_{State::IDLE};
  rclcpp::Time stop_start_;
  double pivot_dir_{0.0};

  // pivot latch / skip-rebrake
  rclcpp::Time last_pivot_cmd_t_, last_pivot_exit_t_;
  double last_pivot_dir_{0.0}, last_exit_pivot_dir_{0.0}, last_pivot_target_{0.0};
  bool   have_pivot_cmd_{false}, have_pivot_exit_{false};

  // arc steer slew
  double arc_steer_out_{0.0};
  rclcpp::Time arc_prev_t_;
  bool   arc_prev_set_{false};

  // odom-derived
  double last_yaw_{0.0}, prev_yaw_{0.0}, prev_px_{0.0}, prev_py_{0.0};
  double yaw_rate_meas_{0.0}, fwd_speed_meas_{0.0};
  rclcpp::Time last_odom_t_;
  bool   have_odom_{false}, have_prev_odom_{false};

  // pivot loop
  double pivot_dev_{100.0};
  rclcpp::Time pivot_win_start_;
  double pivot_win_yaw_{0.0};
  bool   pivot_win_set_{false};

  rclcpp::Publisher<mavros_msgs::msg::OverrideRCIn>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr      state_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelRcBridge>());
  rclcpp::shutdown();
  return 0;
}