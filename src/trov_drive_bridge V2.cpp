// cmd_vel_rc_bridge.cpp
//
// Nav2 cmd_vel  ->  ArduPilot Rover RC override (native skid-steer mixer).
//
// Subscribes to a Twist (default /cmd_vel_safe, i.e. AFTER velocity_smoother
// + collision_monitor) and publishes mavros_msgs/OverrideRCIn to
// /mavros/rc/override, driving ONLY the steering channel (RC1) and throttle
// channel (RC3). Every other channel (incl. the arm channel, RC6) is published
// as 0 = "use real RC", so the transmitter keeps arm/mode and regains
// steer+throttle the instant autonomy stops.
//
// Behaviour (agreed design):
//   * Gate:  fresh cmd_vel within watchdog -> bridge drives RC1/RC3.
//            no fresh cmd_vel             -> release ALL channels (remote through).
//   * Nav commands ~0                     -> hold neutral (1501, brake band), stay put.
//   * DRIVE : gentle forward arcs. Steer+throttle BLENDED (native mixer does
//             left=thr+steer, right=thr-steer). Inner wheel never asked to reverse.
//   * STOP  : turn too tight (radius < r_enter_pivot) -> snap neutral, hold brake_time_s.
//             (this neutral dwell also clears your brake-first ESC lockout)
//   * PIVOT : rotate in place at a fixed slow rate until Nav2 wants to move again.
//   * Comfort cap: |throttle| + |steer| is limited, scaling BOTH down together,
//                  so sharpening a turn never speeds you up.
//
// Left/right treated identically; tight turns become slow in-place pivots.
// Most numeric params are live-tunable: ros2 param set /cmd_vel_rc_bridge <name> <val>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mavros_msgs/msg/override_rc_in.hpp>
#include <mavros_msgs/msg/state.hpp>

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
}  // namespace

class CmdVelRcBridge : public rclcpp::Node {
public:
  CmdVelRcBridge() : Node("cmd_vel_rc_bridge") {
    // ---- topics / timing ----
    input_topic_  = declare_parameter<std::string>("input_topic", "/cmd_vel_safe");
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
    pivot_pwm_    = declare_parameter<double>("pivot_steer_pwm", 100.0); // deviation while pivoting
    comfort_cap_  = declare_parameter<double>("comfort_cap", 1.0);       // cap on |thr_n|+|steer_n|
    pwm_min_      = declare_parameter<double>("pwm_min", 1000.0);
    pwm_max_      = declare_parameter<double>("pwm_max", 2000.0);

    // ---- cmd_vel -> normalized mapping ----
    v_scale_ = declare_parameter<double>("v_scale", 0.5);  // m/s  -> throttle_n = 1.0
    w_scale_ = declare_parameter<double>("w_scale", 1.0);  // rad/s -> steer_n   = 1.0

    // ---- sign conventions (flip on the bench if it drives/turns wrong) ----
    thr_sign_   = declare_parameter<int>("thr_sign", 1)   >= 0 ? 1.0 : -1.0;
    steer_sign_ = declare_parameter<int>("steer_sign", 1) >= 0 ? 1.0 : -1.0;

    // ---- state machine thresholds ----
    enable_pivot_ = declare_parameter<bool>("enable_pivot", true);
    v_eps_        = declare_parameter<double>("v_eps", 0.03);
    w_eps_        = declare_parameter<double>("w_eps", 0.05);
    r_enter_      = declare_parameter<double>("r_enter_pivot", 0.5);
    r_exit_       = declare_parameter<double>("r_exit_pivot", 0.7);
    brake_time_s_ = declare_parameter<double>("brake_time_s", 1.0);

    // ---- safety gate ----
    require_armed_ = declare_parameter<bool>("require_armed", true);

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
      "Reminder: vehicle must be ARMED + in MANUAL for override to drive; "
      "SYSID_MYGCS must match MAVROS system_id (yours=1) or override is silently ignored.");
  }

private:
  enum class State { IDLE, DRIVE, STOP, PIVOT, HOLD };

  // ---------- callbacks ----------
  void on_cmd(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mtx_);
    last_v_ = msg->linear.x;
    last_w_ = msg->angular.z;
    last_cmd_t_ = now();
    have_cmd_ = true;
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

  // Live-tune the numeric knobs without a restart.
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
      else if (n == "enable_pivot")    enable_pivot_= p.as_bool();
      else if (n == "v_eps")           v_eps_       = p.as_double();
      else if (n == "w_eps")           w_eps_       = p.as_double();
      else if (n == "r_enter_pivot")   r_enter_     = p.as_double();
      else if (n == "r_exit_pivot")    r_exit_      = p.as_double();
      else if (n == "brake_time_s")    brake_time_s_= p.as_double();
      else if (n == "watchdog_s")      watchdog_s_  = p.as_double();
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
      mc.channels[arm_idx_] = RC_RELEASE;  // guarantee: never touch arm
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

    // watchdog: nav silent -> hand everything back to the remote
    const double since = have_cmd_ ? (t - last_cmd_t_).seconds()
                                   : std::numeric_limits<double>::infinity();
    if (since > watchdog_s_) {
      if (state_ != State::IDLE)
        RCLCPP_INFO(get_logger(), "no cmd_vel (%.0f ms) -> releasing to remote (IDLE)",
          std::isinf(since) ? -1.0 : since * 1e3);
      state_ = State::IDLE;
      publish_release();
      return;
    }

    const double v = last_v_, w = last_w_;

    // nav commands a stop -> hold position in the brake band
    if (std::fabs(v) < v_eps_ && std::fabs(w) < w_eps_) {
      state_ = State::HOLD;
      publish_neutral();
      log_throttled(v, w, std::numeric_limits<double>::infinity());
      return;
    }

    const double R = (std::fabs(w) > w_eps_)
                       ? std::fabs(v) / std::fabs(w)
                       : std::numeric_limits<double>::infinity();

    if (enable_pivot_) {
      if ((state_ == State::IDLE || state_ == State::DRIVE || state_ == State::HOLD)
          && R < r_enter_) {
        state_ = State::STOP;
        stop_start_ = t;
        pivot_dir_ = sgn(steer_sign_ * w);
      }

      if (state_ == State::STOP) {
        publish_neutral();
        if ((t - stop_start_).seconds() >= brake_time_s_)
          state_ = State::PIVOT;
        log_throttled(v, w, R);
        return;
      }

      if (state_ == State::PIVOT) {
        if (R > r_exit_) {
          state_ = State::DRIVE;  // Nav2 wants to move forward again
        } else {
          publish_pair(neutral_ + pivot_dir_ * pivot_pwm_, neutral_);
          log_throttled(v, w, R);
          return;
        }
      }
    }

    // DRIVE: gentle blended arc
    state_ = State::DRIVE;
    double thr_n   = clampd(thr_sign_   * v / v_scale_, -1.0, 1.0);
    double steer_n = clampd(steer_sign_ * w / w_scale_, -1.0, 1.0);

    // keep the inner wheel from being asked to reverse (that's pivot territory)
    if (enable_pivot_ && std::fabs(thr_n) > 1e-3)
      steer_n = sgn(steer_n) * std::min(std::fabs(steer_n), std::fabs(thr_n));

    // comfort cap: scale throttle AND steer together -> sharper is never faster
    const double s = std::fabs(thr_n) + std::fabs(steer_n);
    if (s > comfort_cap_ && s > EPS) {
      const double k = comfort_cap_ / s;
      thr_n *= k;
      steer_n *= k;
    }

    publish_pair(neutral_ + steer_n * steer_span_, neutral_ + thr_n * thr_span_);
    log_throttled(v, w, R);
  }

  void log_throttled(double v, double w, double R) {
    const char * s = state_name();
    if (std::isinf(R))
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] v=%+.2f w=%+.2f R=inf", s, v, w);
    else
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "[%s] v=%+.2f w=%+.2f R=%.2fm", s, v, w, R);
  }

  const char * state_name() const {
    switch (state_) {
      case State::IDLE:  return "IDLE";
      case State::DRIVE: return "DRIVE";
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
  double v_scale_{0.5}, w_scale_{1.0}, thr_sign_{1.0}, steer_sign_{1.0};
  bool   enable_pivot_{true};
  double v_eps_{0.03}, w_eps_{0.05}, r_enter_{0.5}, r_exit_{0.7}, brake_time_s_{1.0};
  bool   require_armed_{true};

  std::mutex mtx_;
  double last_v_{0.0}, last_w_{0.0};
  rclcpp::Time last_cmd_t_;
  bool have_cmd_{false};
  bool is_armed_{false}, was_armed_{false};

  State state_{State::IDLE};
  rclcpp::Time stop_start_;
  double pivot_dir_{0.0};

  rclcpp::Publisher<mavros_msgs::msg::OverrideRCIn>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr      state_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelRcBridge>());
  rclcpp::shutdown();
  return 0;
}