// trov_drive_bridge_outdoor.cpp
//
// Bridge: /cmd_vel_smoothed (geometry_msgs/Twist, from nav2)
//      -> /mavros/rc/override (mavros_msgs/OverrideRCIn)
//
// ── WHAT CHANGED FROM THE PREVIOUS VERSION ────────────────────────────────
//
// 1. VELOCITY-PROPORTIONAL THROTTLE (was: boolean "lin > 0 -> 1530")
//    linear.x is now mapped onto a PWM range instead of being used as an
//    on/off flag. This is what makes the nav2 YAML actually matter:
//
//        pwm = throttle_min_move + (lin / v_max_scale) * (throttle_max - throttle_min_move)
//        i.e. 0.0 m/s -> neutral, 0.3 m/s -> ~1520, 0.6 m/s -> 1530
//
//    OPEN LOOP. This is proportional to the *command*, not to measured
//    ground speed. Nothing here closes the loop against odom. If 1530
//    physically produces 0.9 m/s while nav2 asked for 0.6, the robot does
//    0.9 and this node neither knows nor cares.
//
//    NOTE ON DYNAMIC RANGE: 1510..1530 is only 20 PWM total, and regulated
//    pure pursuit spends most of its life scaling *down* (curvature, cost,
//    goal approach). Expect to sit around 1512..1520 in practice and to
//    rarely touch 1530. If the robot feels sluggish after this change, the
//    fix is raising throttle_max_pwm, NOT raising desired_linear_vel.
//
// 2. THROTTLE SLEW LIMITER (ramp up / ramp down)
//    All *comfort* throttle output now passes through slew_throttle(), which
//    moves ramp_step_pwm every ramp_interval_ms toward the target.
//    Default 3 PWM / 300 ms => 1510..1530 traversed in ~2.1s.
//
//    Steer is deliberately NOT ramped (explicit instruction) — pivots are
//    stop-and-turn, so a steer ramp would only soften the start of a pivot.
//
//    DEAD-BAND SHORTCUT: 1501..1510 produces no motion at all, so ramping
//    through it would waste ~1s doing nothing. Going up we jump straight to
//    throttle_min_move_pwm and ramp from there; coming down we ramp to
//    throttle_min_move_pwm and then drop to neutral in one step. Both jumps
//    are by definition imperceptible.
//
//    BYPASS LIST — these paths are INSTANT and reset the limiter, because a
//    safety stop must never be slowed down by a comfort feature:
//        no rc_in / no scan / disarmed / arm-warmup /
//        collision stop / stall zeroing / STOPPING settle / PIVOT
//
// 3. GRACEFUL RELEASE (new "releasing" phase)
//    Previously: cmd_vel goes silent -> 300ms -> all channels released
//    instantly, throttle vanishes in one tick. Now the bridge ramps the
//    throttle down to neutral FIRST, then releases.
//
//    COST, be aware of it: RC hand-back is delayed by the ramp-down
//    duration (~2s at the default rates) on top of cmd_vel_timeout_ms. If
//    you flip to manual mid-navigation the sticks take over ~2.3s later.
//    Accel and decel rates are shared (explicit instruction); if that delay
//    turns out to be annoying in the field, split ramp_step_pwm into
//    separate up/down params — the limiter already takes a direction.
//
//    If the last output was already neutral, release is immediate (no
//    pointless 2s hold). If a fresh cmd_vel arrives mid-ramp-down, the
//    ramp is ABORTED and driving resumes from the current PWM — no lurch,
//    no restart from neutral.
//
// 4. ARM-WARMUP (ported from line_follower_rc_bridge)
//    On every disarmed->armed edge, both channels are held at neutral for
//    arm_warmup_sec before any cmd_vel conversion resumes, so the robot
//    never lurches the instant it is armed. Arm-edge detection now lives in
//    control_loop(), NOT in state_callback() — the callback only records
//    is_armed_. Having the edge in two places was asking for them to drift.
//
//    After warmup completes, have_cmd_vel_ is cleared so the watchdog forces
//    a wait for a genuinely FRESH message from nav2 (explicit instruction).
//    A stale pre-disarm command can never drive the robot.
//
// 5. SLOWING no longer uses a fixed creep constant — with the velocity map
//    in place, SLOW_THROTTLE_ was redundant. SLOWING now maps velocity the
//    same way STRAIGHT does; nav2 is already scaling the command down.
//
// 6. Stall detection excludes the releasing phase, or every single
//    ramp-down would look like "commanding motion but not moving".
//
// The ARM channel (RC6) is NEVER overridden — always released (0), so
// arm/disarm stays 100% with the physical remote.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <mavros_msgs/msg/override_rc_in.hpp>
#include <mavros_msgs/msg/rc_in.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>

enum class DriveState { STRAIGHT, SLOWING, STOPPING, PIVOT, STALLED };

struct Point2D { double x; double y; };

class TrovBridgeDrive : public rclcpp::Node {
public:
  TrovBridgeDrive() : Node("trov_drive_bridge") {
    sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_smoothed", 10,
      std::bind(&TrovBridgeDrive::cmd_vel_callback, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10,
      std::bind(&TrovBridgeDrive::odom_callback, this, std::placeholders::_1));

    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", 10,
      std::bind(&TrovBridgeDrive::state_callback, this, std::placeholders::_1));

    rc_in_sub_ = create_subscription<mavros_msgs::msg::RCIn>(
      "/mavros/rc/in", 10,
      std::bind(&TrovBridgeDrive::rc_in_callback, this, std::placeholders::_1));

    // ── Throttle mapping + ramp params ──────────────────────────────────
    // All runtime-tunable so you can calibrate on the bench without a
    // rebuild. The constexpr THROTTLE_MAX_/THROTTLE_MIN_ below remain as
    // absolute hard limits that these params are clamped into — a typo in
    // a param file must not be able to command full throttle or reverse.
    throttle_min_move_pwm_ = this->declare_parameter<double>(
      // Measured on the bench: robot does not move below this.
      "throttle_min_move_pwm", 1510.0);
    throttle_max_pwm_ = this->declare_parameter<double>(
      // Operating ceiling. Raise THIS (not desired_linear_vel) if too slow.
      "throttle_max_pwm", 1560.0);
    v_max_scale_ = this->declare_parameter<double>(
      // The linear.x that maps to throttle_max_pwm_. Should track
      // controller_server FollowPath desired_linear_vel in your nav2 YAML.
      // If they disagree, the robot simply never reaches full scale (if this
      // is higher) or saturates early (if lower). Not dangerous, just
      // surprising — keep them in sync.
      "v_max_scale", 0.6);
    ramp_step_pwm_ = this->declare_parameter<double>(
      "ramp_step_pwm", 3.0);
    ramp_interval_ms_ = this->declare_parameter<int>(
      // 3 PWM / 300ms over a 20 PWM span = 7 steps = ~2.1s full ramp.
      // NOTE: the control loop is 20Hz (50ms), so this MUST be a multiple-ish
      // of 50ms to be meaningful — stepping once per tick would traverse the
      // whole range in 200ms, which is not a ramp at all.
      "ramp_interval_ms", 300);
    arm_warmup_sec_ = this->declare_parameter<double>(
      "arm_warmup_sec", 2.0);

    // Sanity-clamp the params into the compiled hard limits.
    throttle_min_move_pwm_ =
      std::clamp(throttle_min_move_pwm_, THROTTLE_MIN_, THROTTLE_MAX_);
    throttle_max_pwm_ =
      std::clamp(throttle_max_pwm_, throttle_min_move_pwm_, THROTTLE_MAX_);
    if (v_max_scale_ <= 1e-6) {
      RCLCPP_ERROR(get_logger(),
        "v_max_scale must be > 0 (got %.3f) — forcing 0.6.", v_max_scale_);
      v_max_scale_ = 0.6;
    }
    if (ramp_step_pwm_ < 1.0) {
      RCLCPP_ERROR(get_logger(),
        "ramp_step_pwm must be >= 1 (got %.1f) — forcing 3.", ramp_step_pwm_);
      ramp_step_pwm_ = 3.0;
    }

    // ── Collision self-check setup ──────────────────────────────────────
    // collision_monitor's own state_topic feature isn't available on this
    // system's installed nav2_collision_monitor build (state_topic reports
    // "Parameter not set", no publisher created — confirmed via ros2 param
    // get / ros2 node info, even after upgrading to the latest available
    // Humble package). So this replicates the same PolygonStop check
    // directly against /scan, independent of collision_monitor entirely.
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    this->declare_parameter<std::vector<double>>(
      // Forward-only lookout zone: flush against the robot's front edge
      // (x=1.0, matching footprint exactly — never overlaps the robot
      // body), same width as the footprint (y -0.7 to 0.7), extending
      // 0.8m forward (ASSUMPTION — not a computed stopping distance,
      // just a starting point; tune via this param without recompiling).
      // Footprint for reference: [[1.0,0.7],[1.0,-0.7],[-0.7,-0.7],[-0.7,0.7]]
      //"collision_polygon", {1.0, 1.3,  1.0, 0.9,  -0.7, 0.9, -0.7, 1.3});
      "collision_polygon", {1.13, 0.7, 1.5, 0.7, 1.5, -0.7, 1.13, -0.7});

    this->declare_parameter<int>("collision_min_points", 12);
    this->declare_parameter<std::string>("collision_base_frame", "base_link");
    this->declare_parameter<std::string>("scan_topic", "/scan");
    // Enforcement on/off — detection, logging, and the RViz marker keep
    // running regardless of this; only the "force neutral" gate in
    // control_loop() respects it. The topic below overrides this at runtime.
    this->declare_parameter<bool>("collision_check_enabled", false);

    std::vector<double> polygon_flat =
      this->get_parameter("collision_polygon").as_double_array();
    collision_min_points_   = this->get_parameter("collision_min_points").as_int();
    collision_base_frame_   = this->get_parameter("collision_base_frame").as_string();
    scan_topic_             = this->get_parameter("scan_topic").as_string();
    collision_check_enabled_ = this->get_parameter("collision_check_enabled").as_bool();

    if (polygon_flat.size() < 6 || polygon_flat.size() % 2 != 0) {
      RCLCPP_ERROR(get_logger(),
        "collision_polygon parameter malformed (need >=3 points as flat "
        "x,y pairs) — falling back to hardcoded forward-lookout default.");
      polygon_flat = {1.8, 0.7,  1.8, -0.7,  1.0, -0.7,  1.0, 0.7};
    }
    for (size_t i = 0; i + 1 < polygon_flat.size(); i += 2) {
      collision_polygon_.push_back({polygon_flat[i], polygon_flat[i + 1]});
    }

    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_, rclcpp::SensorDataQoS(),
      std::bind(&TrovBridgeDrive::scan_callback, this, std::placeholders::_1));

    // Runtime toggle — publish std_msgs/Bool here to enable/disable
    // enforcement live, without restarting the node.
    collision_check_enable_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/collision_check_enable", 10,
      std::bind(&TrovBridgeDrive::collision_check_enable_callback, this, std::placeholders::_1));

    // RViz visualization of the collision self-check zone — republished
    // from our own collision_polygon_ param (not collision_monitor's
    // /polygon_stop), so it reflects exactly what this node is actually
    // checking against.
    polygon_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      "/collision_check_polygon", 10);

    pub_ = create_publisher<mavros_msgs::msg::OverrideRCIn>(
      "/mavros/rc/override", 10);

    // Drive/publish decisions run on this fixed-rate timer, NOT on the
    // cmd_vel subscription callback — this is what lets us detect
    // "/cmd_vel_smoothed has gone silent" (nav stack idle) instead of just
    // never running again if the topic stops. It is also what paces the
    // slew limiter, which needs ticks even when no command arrives.
    control_timer_ = create_wall_timer(
      std::chrono::milliseconds(CONTROL_PERIOD_MS_),
      std::bind(&TrovBridgeDrive::control_loop, this));

    slew_last_step_ = now();

    // ── BUILD MARKER — confirm this matches your latest build before
    //    trusting any other log line below. ─────────────────────────────
    RCLCPP_WARN(get_logger(),
      "=== TrovBridgeDrive BUILD: %s %s ===", __DATE__, __TIME__);

    RCLCPP_INFO(get_logger(),
      "TROV Drive Bridge started — steer=RC%zu(neutral=%.0f right_base=%.0f "
      "left_base=%.0f min=%.0f max=%.0f NOT_RAMPED) "
      "throttle=RC%zu(neutral=%.0f min_move=%.0f max=%.0f hard_ceiling=%.0f "
      "no_reverse=true) "
      "arm=RC%zu(ALWAYS RELEASED) cmd_vel_timeout_ms=%d control_period_ms=%d "
      "ang_threshold=%.3f ang_slowzone=%.3f "
      "stop_duration_ms=%d stall_check_ms=%d stall_dist=%.4f stall_zero_ms=%d",
      STEER_CH_ + 1, NEUTRAL_PWM_, PIVOT_RIGHT_BASE_, PIVOT_LEFT_BASE_,
      PIVOT_MIN_, PIVOT_MAX_,
      THROTTLE_CH_ + 1, NEUTRAL_PWM_, throttle_min_move_pwm_,
      throttle_max_pwm_, THROTTLE_MAX_,
      ARM_CH_ + 1, CMD_VEL_TIMEOUT_MS_, CONTROL_PERIOD_MS_,
      ANG_THRESHOLD_, ANG_SLOW_ZONE_,
      STOP_DURATION_MS_, STALL_CHECK_MS_, STALL_DIST_THRESHOLD_,
      STALL_ZERO_MS_);

    {
      const double span  = throttle_max_pwm_ - throttle_min_move_pwm_;
      const double steps = (ramp_step_pwm_ > 0.0) ? (span / ramp_step_pwm_) : 0.0;
      RCLCPP_WARN(get_logger(),
        "THROTTLE MAP: lin 0..%.2f m/s -> PWM %.0f..%.0f (open loop, "
        "command-proportional, NOT odom-calibrated). "
        "RAMP: %.0f PWM every %dms => ~%.2fs full ramp (%.0f steps), "
        "same rate up and down. Arm-warmup=%.1fs. "
        "Steer is NOT ramped by design.",
        v_max_scale_, throttle_min_move_pwm_, throttle_max_pwm_,
        ramp_step_pwm_, ramp_interval_ms_,
        steps * (ramp_interval_ms_ / 1000.0), steps, arm_warmup_sec_);

      if (DEADZONE_ >= 0.15 * v_max_scale_) {
        RCLCPP_WARN(get_logger(),
          "NOTE: cmd_vel deadzone (%.3f) is %.0f%% of v_max_scale (%.2f) — "
          "the bottom of the velocity map is unreachable and low-speed "
          "approach commands will snap straight to neutral. Lower DEADZONE_ "
          "if goal approach feels abrupt.",
          DEADZONE_, 100.0 * DEADZONE_ / v_max_scale_, v_max_scale_);
      }
    }

    {
      std::ostringstream poly_oss;
      for (size_t i = 0; i < collision_polygon_.size(); ++i) {
        poly_oss << "(" << collision_polygon_[i].x << "," << collision_polygon_[i].y << ")";
        if (i + 1 < collision_polygon_.size()) poly_oss << " ";
      }
      RCLCPP_WARN(get_logger(),
        "Collision self-check ACTIVE (collision_monitor's state_topic is NOT "
        "available on this system) — scan_topic=%s base_frame=%s "
        "min_points=%d polygon=[%s] enforcement_enabled=%s "
        "(toggle live via /collision_check_enable, std_msgs/Bool)",
        scan_topic_.c_str(), collision_base_frame_.c_str(),
        collision_min_points_, poly_oss.str().c_str(),
        collision_check_enabled_ ? "true" : "false");
    }
  }

private:
  // ── RC channel map (0-indexed; RC1=0, RC3=2, RC6=5) ────────────────────
  static constexpr size_t STEER_CH_    = 0;  // RC1 - rotation
  static constexpr size_t THROTTLE_CH_ = 2;  // RC3 - forward drive
  static constexpr size_t ARM_CH_      = 5;  // RC6 - arm/disarm (NEVER overridden)
  static constexpr uint16_t RC_RELEASE_ = 0; // 0 = "don't touch, use real RC input"

  // ── Shared neutral ───────────────────────────────────────────────────
  static constexpr double NEUTRAL_PWM_ = 1501.0;

  // ── Throttle (RC3) — forward only, never reverse ────────────────────
  // These two are ABSOLUTE compiled limits. The tunable params
  // throttle_min_move_pwm_ / throttle_max_pwm_ are clamped into this range
  // at startup, so no param file can command reverse or overspeed.
  static constexpr double THROTTLE_MAX_ = 1600.0;        // HARD CEILING
  static constexpr double THROTTLE_MIN_ = NEUTRAL_PWM_;  // HARD FLOOR — no reverse, ever
  static constexpr int    STRAIGHT_CHECK_MS_       = 1500; // ASSUMPTION: tune me
  static constexpr double STRAIGHT_DIST_THRESHOLD_ = 0.05; // meters, ASSUMPTION: tune me

  // ── Steer / rotation (RC1) — NOT slew-limited (explicit instruction) ──
  // ASSUMPTION: ang > 0 (ROS REP-103, angular.z counter-clockwise) = LEFT.
  // Verify this matches your hardware's actual turn direction on the bench.
  static constexpr double PIVOT_RIGHT_BASE_ = 1670.0; // output turning right
  static constexpr double PIVOT_LEFT_BASE_  = 1310.0; // output turning left
  static constexpr double PIVOT_MIN_        = 1100.0; // HARD FLOOR
  static constexpr double PIVOT_MAX_        = 1800.0; // HARD CEILING
  static constexpr int    PIVOT_CHECK_MS_      = 1500;
  static constexpr double PIVOT_ROT_THRESHOLD_ = 0.5;  // radians
  // ang must stay continuously ABOVE threshold this long before PIVOT is
  // entered, and continuously BELOW threshold this long before PIVOT is
  // exited — same debounce window used symmetrically both ways.
  // ASSUMPTION, tune me.
  static constexpr int    PIVOT_HOLD_MS_ = 400;

  // ── cmd_vel watchdog ─────────────────────────────────────────────────
  // If no /cmd_vel_smoothed message arrives within this window the nav
  // stack is considered idle/dead. NOTE: as of the graceful-release change
  // this no longer releases immediately — it starts a ramp-down first, so
  // real hand-back time is CMD_VEL_TIMEOUT_MS_ + ramp-down duration.
  static constexpr int    CMD_VEL_TIMEOUT_MS_ = 300;
  static constexpr int    CONTROL_PERIOD_MS_  = 50; // 20Hz control/publish loop

  // ── Drive constants ────────────────────────────────────────────────────
  // DEADZONE_ applies to BOTH lin and ang. On the lin side it now also sets
  // the lowest reachable point of the velocity map — see the startup warning.
  static constexpr double DEADZONE_         = 0.09;
  static constexpr double ANG_THRESHOLD_    = 0.3;
  static constexpr double ANG_SLOW_ZONE_    = 0.2;
  static constexpr int    STOP_DURATION_MS_ = 2000;

  // ── Stall-detection constants (full stop, distinct from throttle ramp) ─
  static constexpr int    STALL_CHECK_MS_       = 7000;
  static constexpr double STALL_DIST_THRESHOLD_ = 0.02;
  static constexpr int    STALL_ZERO_MS_        = 1000;

  // ── Tunable params ─────────────────────────────────────────────────────
  double throttle_min_move_pwm_ = 1510.0;
  double throttle_max_pwm_      = 1560.0;
  double v_max_scale_           = 0.6;
  double ramp_step_pwm_         = 3.0;
  int    ramp_interval_ms_      = 300;
  double arm_warmup_sec_        = 2.0;

  // ── Runtime state ──────────────────────────────────────────────────────
  DriveState   state_    = DriveState::STRAIGHT;
  rclcpp::Time stop_start_;
  double       last_ang_ = 0.0;

  bool is_armed_  = false;
  bool was_armed_ = false;   // owned by control_loop(), NOT state_callback()

  // Arm-warmup
  bool         warmup_active_ = false;
  rclcpp::Time warmup_start_;

  // Graceful release (ramp throttle to neutral, THEN hand back to RC)
  bool releasing_ = false;

  // Throttle slew limiter
  double       slew_current_ = NEUTRAL_PWM_;
  rclcpp::Time slew_last_step_;

  rclcpp::Time stall_window_start_;
  double       stall_window_x_    = 0.0;
  double       stall_window_y_    = 0.0;
  bool         stall_window_set_  = false;
  rclcpp::Time stall_zero_start_;
  bool         output_is_nonzero_ = false;

  // Pivot (rotation) progress tracking
  double       current_pivot_output_ = NEUTRAL_PWM_;
  rclcpp::Time pivot_check_start_;
  double       pivot_start_yaw_      = 0.0;
  bool         pivot_check_set_      = false;
  bool         pivot_exit_pending_   = false;
  rclcpp::Time pivot_exit_start_;
  bool         pivot_entry_pending_   = false;
  rclcpp::Time pivot_entry_start_;
  bool         pivot_entry_was_left_  = false;

  // Straight progress tracking
  rclcpp::Time straight_check_start_;
  double       straight_start_x_ = 0.0;
  double       straight_start_y_ = 0.0;
  bool         straight_check_set_ = false;

  // Latest odom
  double odom_x_   = 0.0;
  double odom_y_   = 0.0;
  double odom_yaw_ = 0.0;
  bool   odom_ok_  = false;

  // Latest cmd_vel (consumed by the timer-driven control loop, not the
  // subscription callback itself — see control_loop())
  bool                                      have_cmd_vel_ = false;
  rclcpp::Time                              last_cmd_vel_time_;
  geometry_msgs::msg::Twist::SharedPtr      last_cmd_vel_msg_;

  // RC input bookkeeping (logging / arm-channel truth only — no longer used
  // for manual-override detection, see conversation history for why)
  bool         rc_in_ok_            = false;
  size_t       rc_in_channel_count_ = 0;
  uint16_t     last_rc_in_arm_raw_  = 0;

  // Collision self-check state (scan-based, see scan_callback()).
  // Defaults to "clear" — flips true only on a confirmed detection.
  bool   collision_stop_active_  = false;
  int    last_points_inside_     = 0;
  bool   scan_ok_                = false;
  bool   collision_check_enabled_ = false;

  // Collision self-check configuration (ROS params, see constructor)
  std::vector<Point2D> collision_polygon_;
  int                   collision_min_points_ = 9;
  std::string           collision_base_frame_ = "base_link";
  std::string           scan_topic_           = "/scan";
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_check_enable_sub_;

  // ══════════════════════════════════════════════════════════════════════
  //  THROTTLE MAPPING + SLEW LIMITER
  // ══════════════════════════════════════════════════════════════════════

  // Map a commanded linear.x onto the throttle PWM range. Open loop.
  double throttle_pwm_for(double lin) const {
    if (lin <= 0.0) return NEUTRAL_PWM_;
    const double f = std::clamp(lin / v_max_scale_, 0.0, 1.0);
    const double pwm =
      throttle_min_move_pwm_ + f * (throttle_max_pwm_ - throttle_min_move_pwm_);
    return std::clamp(pwm, THROTTLE_MIN_, THROTTLE_MAX_);
  }

  // Force the limiter to a value without ramping. MUST be called by every
  // instant/bypass path, otherwise the next ramp resumes from a stale PWM
  // that no longer reflects what the ESC is actually seeing.
  void reset_slew(double to = NEUTRAL_PWM_) {
    slew_current_   = std::clamp(to, THROTTLE_MIN_, THROTTLE_MAX_);
    slew_last_step_ = now();
  }

  // Move slew_current_ one step toward `target`, no faster than
  // ramp_step_pwm_ per ramp_interval_ms_. Returns the value to publish.
  //
  // The 1501..throttle_min_move_pwm_ band is skipped in a single jump in
  // both directions — no motion happens there, so ramping through it is
  // pure dead time.
  double slew_throttle(double target) {
    target = std::clamp(target, THROTTLE_MIN_, THROTTLE_MAX_);

    if (std::fabs(target - slew_current_) < 0.5) {
      slew_current_ = target;
      return slew_current_;
    }

    // Dead-band shortcut, going UP: jump straight onto the first PWM that
    // actually produces motion, then ramp normally from there.
    if (target > slew_current_ && slew_current_ < throttle_min_move_pwm_) {
      slew_current_   = std::min(throttle_min_move_pwm_, target);
      slew_last_step_ = now();
      return slew_current_;
    }

    // Dead-band shortcut, going DOWN: once we're at the bottom of the
    // moving range and being asked to stop, drop to neutral in one step.
    if (target < slew_current_ &&
        slew_current_ <= throttle_min_move_pwm_ + 0.5 &&
        target <= throttle_min_move_pwm_) {
      slew_current_   = std::max(target, NEUTRAL_PWM_);
      slew_last_step_ = now();
      return slew_current_;
    }

    const auto since_step_ms =
      (now() - slew_last_step_).nanoseconds() / 1'000'000;
    if (since_step_ms < ramp_interval_ms_) {
      return slew_current_;  // holding between steps
    }

    slew_last_step_ = now();
    if (target > slew_current_) {
      slew_current_ = std::min(target, slew_current_ + ramp_step_pwm_);
    } else {
      slew_current_ = std::max(target, slew_current_ - ramp_step_pwm_);
    }
    return slew_current_;
  }

  bool slew_at_neutral() const {
    return slew_current_ <= NEUTRAL_PWM_ + 0.5;
  }

  // ── Arm state callback — records the flag ONLY. Edge detection lives in
  //    control_loop() so the warmup timer and the edge can't drift apart. ─
  void state_callback(const mavros_msgs::msg::State::SharedPtr msg) {
    is_armed_ = msg->armed;
  }

  // ── cmd_vel subscription — just stores the latest message. ─────────────
  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    last_cmd_vel_msg_  = msg;
    last_cmd_vel_time_ = now();
    have_cmd_vel_      = true;
  }

  // ── Odom callback ──────────────────────────────────────────────────────
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;

    const auto & q = msg->pose.pose.orientation;
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    tf2::Matrix3x3 m(tf_q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    odom_yaw_ = yaw;

    odom_ok_ = true;
  }

  // ── Collision check enable/disable toggle — runtime override via topic. ─
  void collision_check_enable_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data != collision_check_enabled_) {
      RCLCPP_WARN(get_logger(),
        "COLLISION CHECK enforcement toggled -> %s (via /collision_check_enable)",
        msg->data ? "ENABLED" : "DISABLED");
    }
    collision_check_enabled_ = msg->data;
  }

  void set_collision_stop(bool stop, int points_inside, const char * reason) {
    if (stop != collision_stop_active_) {
      RCLCPP_WARN(get_logger(),
        "COLLISION CHECK: state changed -> %s (reason=%s, points_inside=%d, "
        "min_points=%d)",
        stop ? "STOP" : "CLEAR", reason, points_inside, collision_min_points_);
      reset_straight_state();
      reset_pivot_state();
      reset_stall_window();
    }
    collision_stop_active_ = stop;
    last_points_inside_    = points_inside;
  }

  // Ray-casting point-in-polygon test (even-odd rule). Works for any
  // simple polygon, not just the current rectangular PolygonStop shape.
  bool point_in_polygon(double x, double y) const {
    bool inside = false;
    const size_t n = collision_polygon_.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
      const double xi = collision_polygon_[i].x, yi = collision_polygon_[i].y;
      const double xj = collision_polygon_[j].x, yj = collision_polygon_[j].y;
      const bool intersect = ((yi > y) != (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  void publish_polygon_marker([[maybe_unused]] bool stop_active, const rclcpp::Time & stamp) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = collision_base_frame_;
    marker.header.stamp    = stamp;
    marker.ns     = "collision_self_check";
    marker.id     = 0;
    marker.type   = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f; marker.color.a = 1.0f;

    marker.points.reserve(collision_polygon_.size() + 1);
    for (const auto & pt : collision_polygon_) {
      geometry_msgs::msg::Point p;
      p.x = pt.x; p.y = pt.y; p.z = 0.0;
      marker.points.push_back(p);
    }
    if (!collision_polygon_.empty()) {
      geometry_msgs::msg::Point p;
      p.x = collision_polygon_.front().x;
      p.y = collision_polygon_.front().y;
      p.z = 0.0;
      marker.points.push_back(p);
    }

    polygon_marker_pub_->publish(marker);
  }

  // ── Scan callback — the collision self-check itself. ──────────────────
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    scan_ok_ = true;

    double tx = 0.0, ty = 0.0, cy = 1.0, sy = 0.0; // identity by default

    if (msg->header.frame_id != collision_base_frame_) {
      geometry_msgs::msg::TransformStamped tf_stamped;
      try {
        tf_stamped = tf_buffer_->lookupTransform(
          collision_base_frame_, msg->header.frame_id, msg->header.stamp,
          rclcpp::Duration::from_seconds(0.1));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "COLLISION CHECK: TF lookup %s -> %s failed (%s) — cannot verify "
          "clearance, failing safe to STOP.",
          msg->header.frame_id.c_str(), collision_base_frame_.c_str(), ex.what());
        set_collision_stop(true, -1, "tf_lookup_failed");
        publish_polygon_marker(true, msg->header.stamp);
        return;
      }

      tx = tf_stamped.transform.translation.x;
      ty = tf_stamped.transform.translation.y;
      tf2::Quaternion q(
        tf_stamped.transform.rotation.x, tf_stamped.transform.rotation.y,
        tf_stamped.transform.rotation.z, tf_stamped.transform.rotation.w);
      double roll, pitch, yaw;
      tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
      cy = std::cos(yaw);
      sy = std::sin(yaw);
    }

    int points_inside = 0;
    for (size_t i = 0; i < msg->ranges.size(); ++i) {
      const float range = msg->ranges[i];
      if (!std::isfinite(range)) continue;
      if (range < msg->range_min || range > msg->range_max) continue;

      const double angle = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
      const double lx = range * std::cos(angle);
      const double ly = range * std::sin(angle);

      const double bx = tx + cy * lx - sy * ly;
      const double by = ty + sy * lx + cy * ly;

      if (point_in_polygon(bx, by)) {
        ++points_inside;
        if (points_inside >= collision_min_points_) break;
      }
    }

    const bool now_stop = (points_inside >= collision_min_points_);
    set_collision_stop(now_stop, points_inside, "scan_check");
    publish_polygon_marker(now_stop, msg->header.stamp);
  }

  // ── RC input callback ──────────────────────────────────────────────────
  void rc_in_callback(const mavros_msgs::msg::RCIn::SharedPtr msg) {
    if (rc_in_channel_count_ != msg->channels.size()) {
      rc_in_channel_count_ = msg->channels.size();
      RCLCPP_WARN(get_logger(),
        "/mavros/rc/in reports %zu channels", rc_in_channel_count_);
    }

    if (msg->channels.size() <= std::max(STEER_CH_, THROTTLE_CH_)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "/mavros/rc/in has fewer channels (%zu) than needed — cannot read sticks!",
        msg->channels.size());
      return;
    }
    rc_in_ok_ = true;

    if (msg->channels.size() > ARM_CH_) {
      last_rc_in_arm_raw_ = msg->channels[ARM_CH_];
    }

    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
    //   "[DEBUG rc/in] steer(ch%zu)=%d throttle(ch%zu)=%d arm(ch%zu)=%d "
    //   "(not used for control decisions — see cmd_vel watchdog instead)",
    //   STEER_CH_ + 1, msg->channels[STEER_CH_],
    //   THROTTLE_CH_ + 1, msg->channels[THROTTLE_CH_],
    //   ARM_CH_ + 1, last_rc_in_arm_raw_);
  }

  // ── Helpers ────────────────────────────────────────────────────────────
  void reset_pivot_state() {
    current_pivot_output_ = NEUTRAL_PWM_;
    pivot_check_set_      = false;
    pivot_exit_pending_    = false;
    pivot_entry_pending_   = false;
  }

  void reset_straight_state() {
    straight_check_set_ = false;
  }

  void reset_stall_window() {
    stall_window_set_ = false;
  }

  // Everything that must be forgotten when the chassis stops being ours.
  void reset_all_drive_state() {
    reset_pivot_state();
    reset_straight_state();
    reset_stall_window();
    reset_slew();
    releasing_ = false;
    state_     = DriveState::STRAIGHT;
  }

  mavros_msgs::msg::OverrideRCIn make_release_all() const {
    mavros_msgs::msg::OverrideRCIn mc;
    mc.channels.fill(RC_RELEASE_);
    return mc;
  }

  mavros_msgs::msg::OverrideRCIn make_neutral_both() const {
    mavros_msgs::msg::OverrideRCIn mc = make_release_all();
    mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
    mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
    return mc;
  }

  // ── SINGLE publish path — arm channel guaranteed released here ────────
  void publish_override(mavros_msgs::msg::OverrideRCIn mc, const char * reason) {
    mc.channels[ARM_CH_] = RC_RELEASE_;

    std::ostringstream oss;
    for (size_t i = 0; i < mc.channels.size(); ++i) {
      oss << mc.channels[i];
      if (i + 1 < mc.channels.size()) oss << ",";
    }
    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
    //   "[DEBUG override PUBLISH reason=%s] arm_ch(RC%zu)=%d rc_in_arm_raw=%d "
    //   "slew=%.0f channels=[%s]",
    //   reason, ARM_CH_ + 1, mc.channels[ARM_CH_], last_rc_in_arm_raw_,
    //   slew_current_, oss.str().c_str());

    pub_->publish(mc);
  }

  // ══════════════════════════════════════════════════════════════════════
  //  CONTROL LOOP
  //
  //  PRECEDENCE (order matters, do not shuffle):
  //    rc_in gate -> scan gate -> armed gate -> ARM-WARMUP ->
  //    collision stop -> cmd_vel watchdog / graceful release ->
  //    drive state machine
  //
  //  Warmup sits ahead of everything drive-related so nothing moves in the
  //  first arm_warmup_sec after arming. Collision stop sits ahead of the
  //  watchdog because a detected obstacle must win even if nav2 just went
  //  quiet mid-ramp-down.
  // ══════════════════════════════════════════════════════════════════════
  void control_loop() {
    // ── Gate 1: receiver alive (INSTANT, bypasses ramp) ─────────────────
    if (!rc_in_ok_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No /mavros/rc/in received yet — refusing to drive, releasing all channels.");
      publish_override(make_release_all(), "no_rc_in_yet");
      reset_slew();
      output_is_nonzero_ = false;
      return;
    }

    // ── Gate 2: scan alive, only when enforcement is on (INSTANT) ───────
    if (collision_check_enabled_ && !scan_ok_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No /scan received yet — cannot verify collision clearance, "
        "refusing to drive, releasing all channels.");
      publish_override(make_release_all(), "no_scan_yet");
      reset_slew();
      output_is_nonzero_ = false;
      return;
    }

    // ── Gate 3: armed. Also owns BOTH arm edges. (INSTANT) ──────────────
    if (!is_armed_) {
      if (was_armed_) {
        RCLCPP_WARN(get_logger(),
          "DISARM detected — resetting ramps, slew limiter and stall window.");
      }
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "Vehicle DISARMED — releasing all channels to RC.");
      publish_override(make_release_all(), "disarmed");
      reset_all_drive_state();
      warmup_active_     = false;
      was_armed_         = false;  // next arm retriggers warmup
      output_is_nonzero_ = false;
      return;
    }

    // disarmed -> armed edge
    if (is_armed_ && !was_armed_) {
      was_armed_     = true;
      warmup_active_ = true;
      warmup_start_  = now();
      reset_all_drive_state();
      // Discard anything received before/while disarmed. A stale command
      // must never be what starts the robot moving after an arm.
      have_cmd_vel_ = false;
      RCLCPP_WARN(get_logger(),
        "ARMED — holding neutral for %.1fs arm-warmup before driving.",
        arm_warmup_sec_);
    }

    // ── ARM-WARMUP (INSTANT neutral, bypasses ramp) ─────────────────────
    if (warmup_active_) {
      const double elapsed = (now() - warmup_start_).seconds();
      if (elapsed < arm_warmup_sec_) {
        publish_override(make_neutral_both(), "arm_warmup");
        reset_slew();
        output_is_nonzero_ = false;
        return;
      }
      warmup_active_ = false;
      // Require a genuinely FRESH cmd_vel after warmup — anything that
      // arrived during the hold is discarded too.
      have_cmd_vel_ = false;
      RCLCPP_WARN(get_logger(),
        "Arm-warmup complete — waiting for fresh /cmd_vel_smoothed before driving.");
    }

    // ── Collision self-check hard stop (INSTANT, bypasses ramp) ─────────
    // Takes priority over both the watchdog and the drive state machine.
    // We actively command neutral (not a full release) since this is an
    // autonomy-active safety stop, not a hand-back-to-RC event.
    if (collision_check_enabled_ && collision_stop_active_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "COLLISION STOP active (points_inside=%d, min_points=%d) — "
        "forcing neutral INSTANTLY (ramp bypassed), drive state machine bypassed.",
        last_points_inside_, collision_min_points_);
      publish_override(make_neutral_both(), "collision_stop");
      reset_slew();
      reset_stall_window();
      releasing_ = false;
      output_is_nonzero_ = false;
      return;
    }

    // ── cmd_vel watchdog + GRACEFUL RELEASE ─────────────────────────────
    const auto since_cmd_vel_ms = have_cmd_vel_
      ? (now() - last_cmd_vel_time_).nanoseconds() / 1'000'000
      : std::numeric_limits<long>::max();

    if (!have_cmd_vel_ || since_cmd_vel_ms > CMD_VEL_TIMEOUT_MS_) {
      // Already at neutral? Release immediately — no point holding the
      // channels for 2s to ramp a throttle that is already off.
      if (slew_at_neutral()) {
        if (releasing_) {
          RCLCPP_INFO(get_logger(),
            "Ramp-down complete — releasing all channels to RC.");
          releasing_ = false;
        }
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
          "/cmd_vel_smoothed idle/silent (%s) — released to RC.",
          have_cmd_vel_ ? "timeout" : "never received");
        publish_override(make_release_all(), "cmd_vel_idle");
        reset_all_drive_state();
        output_is_nonzero_ = false;
        return;
      }

      // Still rolling — ramp the throttle down before handing back.
      if (!releasing_) {
        releasing_ = true;
        RCLCPP_WARN(get_logger(),
          "/cmd_vel_smoothed idle — ramping throttle down from %.0f to neutral "
          "before releasing to RC (hand-back delayed by the ramp).",
          slew_current_);
      }

      mavros_msgs::msg::OverrideRCIn mc = make_release_all();
      mc.channels[THROTTLE_CH_] =
        static_cast<uint16_t>(std::lround(slew_throttle(NEUTRAL_PWM_)));
      mc.channels[STEER_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
      publish_override(mc, "releasing_ramp_down");
      // Stall detection must not run here or every ramp-down would look
      // like "commanding motion while not moving".
      reset_stall_window();
      output_is_nonzero_ = false;
      return;
    }

    // Fresh command arrived — abort any pending ramp-down. Driving picks up
    // from the CURRENT slew value, so there is no lurch and no restart from
    // neutral.
    if (releasing_) {
      releasing_ = false;
      RCLCPP_INFO(get_logger(),
        "Fresh /cmd_vel_smoothed during ramp-down — aborting release, "
        "resuming drive from PWM %.0f.", slew_current_);
    }

    const auto & msg = last_cmd_vel_msg_;
    mavros_msgs::msg::OverrideRCIn mc = make_release_all();

    const double lin = apply_deadzone(msg->linear.x);
    const double ang = apply_deadzone(msg->angular.z);

    // ── STALLED hold (INSTANT neutral, bypasses ramp) ───────────────────
    if (state_ == DriveState::STALLED) {
      const auto elapsed_ms = (now() - stall_zero_start_).nanoseconds() / 1'000'000;
      if (elapsed_ms < STALL_ZERO_MS_) {
        publish_override(make_neutral_both(), "stalled");
        reset_slew();
        output_is_nonzero_ = false;
        return;
      }
      RCLCPP_INFO(get_logger(), "-> STALLED complete, resuming normal control");
      state_ = DriveState::STRAIGHT;
      reset_stall_window();
    }

    // ── Normal state transitions ────────────────────────────────────────
    const DriveState prev_state = state_;

    if (state_ == DriveState::PIVOT) {
      // Exiting PIVOT: ang must stay continuously BELOW threshold for
      // PIVOT_HOLD_MS_ before we actually leave. Any single tick back above
      // threshold cancels the pending exit and resets the timer.
      if (std::fabs(ang) > ANG_THRESHOLD_) {
        pivot_exit_pending_ = false;
      } else {
        if (!pivot_exit_pending_) {
          pivot_exit_pending_ = true;
          pivot_exit_start_   = now();
          RCLCPP_INFO(get_logger(),
            "PIVOT: ang dropped below threshold (%.3f) — starting %dms exit hold",
            ang, PIVOT_HOLD_MS_);
        } else {
          const auto exit_ms =
            (now() - pivot_exit_start_).nanoseconds() / 1'000'000;
          if (exit_ms >= PIVOT_HOLD_MS_) {
            state_ = (std::fabs(ang) > ANG_SLOW_ZONE_)
                       ? DriveState::SLOWING : DriveState::STRAIGHT;
            pivot_exit_pending_ = false;
            RCLCPP_INFO(get_logger(),
              "PIVOT: ang stayed below threshold for %ldms — exiting PIVOT",
              static_cast<long>(exit_ms));
          }
        }
      }
    } else if (state_ != DriveState::STOPPING) {
      if (std::fabs(ang) > ANG_THRESHOLD_) {
        // Entering PIVOT: ang must stay continuously ABOVE threshold, in the
        // SAME direction, for PIVOT_HOLD_MS_ before we commit to a turn.
        // Until confirmed, output stays exactly as it already was — which
        // now means the throttle ramp keeps running undisturbed, so a noise
        // spike costs nothing.
        const bool this_tick_left = (ang > 0.0);

        if (!pivot_entry_pending_ || this_tick_left != pivot_entry_was_left_) {
          pivot_entry_pending_  = true;
          pivot_entry_start_    = now();
          pivot_entry_was_left_ = this_tick_left;
          RCLCPP_INFO(get_logger(),
            "ang crossed threshold (%.3f, dir=%s) — starting %dms entry hold",
            ang, this_tick_left ? "LEFT" : "RIGHT", PIVOT_HOLD_MS_);
        } else {
          const auto entry_ms =
            (now() - pivot_entry_start_).nanoseconds() / 1'000'000;
          if (entry_ms >= PIVOT_HOLD_MS_) {
            state_      = DriveState::STOPPING;
            stop_start_ = now();
            last_ang_   = ang;
            pivot_entry_pending_ = false;
            RCLCPP_INFO(get_logger(),
              "ang held above threshold for %ldms — entering STOPPING phase (%dms)",
              static_cast<long>(entry_ms), STOP_DURATION_MS_);
          }
        }
      } else {
        pivot_entry_pending_ = false; // dipped below threshold, reset confirmation
        if (std::fabs(ang) > ANG_SLOW_ZONE_) {
          state_ = DriveState::SLOWING;
        } else {
          state_ = DriveState::STRAIGHT;
        }
      }
    } else {
      last_ang_ = ang;
      const auto elapsed_ms = (now() - stop_start_).nanoseconds() / 1'000'000;
      if (elapsed_ms >= STOP_DURATION_MS_) {
        state_ = DriveState::PIVOT;
        RCLCPP_INFO(get_logger(), "-> STOPPING complete, entering PIVOT");
      }
    }

    if (prev_state == DriveState::PIVOT && state_ != DriveState::PIVOT) {
      reset_pivot_state();
    }
    if (prev_state == DriveState::STRAIGHT && state_ != DriveState::STRAIGHT) {
      reset_straight_state();
    }

    // ── Output ────────────────────────────────────────────────────────────
    switch (state_) {
      case DriveState::STRAIGHT: {
        if (lin > 0.0) {
          const double target = throttle_pwm_for(lin);

          // Progress check — informational only, ramping is not adaptive.
          if (!straight_check_set_) {
            straight_check_start_ = now();
            straight_start_x_     = odom_x_;
            straight_start_y_     = odom_y_;
            straight_check_set_   = true;
          } else if (odom_ok_) {
            const auto window_ms =
              (now() - straight_check_start_).nanoseconds() / 1'000'000;
            if (window_ms >= STRAIGHT_CHECK_MS_) {
              const double dx   = odom_x_ - straight_start_x_;
              const double dy   = odom_y_ - straight_start_y_;
              const double dist = std::sqrt(dx * dx + dy * dy);

              if (dist < STRAIGHT_DIST_THRESHOLD_) {
                RCLCPP_WARN(get_logger(),
                  "STRAIGHT: insufficient progress (dist=%.3fm in %ldms) at "
                  "cmd %.3f m/s -> target %.0f (current %.0f). No auto-ramp-up.",
                  dist, static_cast<long>(window_ms), lin, target, slew_current_);
              } else {
                RCLCPP_INFO(get_logger(),
                  "STRAIGHT: progress OK (dist=%.3fm) at cmd %.3f m/s -> "
                  "target %.0f (current %.0f)",
                  dist, lin, target, slew_current_);
              }
              straight_check_start_ = now();
              straight_start_x_     = odom_x_;
              straight_start_y_     = odom_y_;
            }
          }

          mc.channels[THROTTLE_CH_] =
            static_cast<uint16_t>(std::lround(slew_throttle(target)));
        } else {
          // No reverse, ever. Ramp DOWN to neutral (not a hard snap) — this
          // is the normal "nav2 asked for zero" stop, so it gets the smooth
          // treatment.
          if (lin < 0.0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
              "cmd_vel requested reverse (lin=%.3f) — reverse disabled, "
              "ramping to neutral throttle.", lin);
          }
          reset_straight_state();
          mc.channels[THROTTLE_CH_] =
            static_cast<uint16_t>(std::lround(slew_throttle(NEUTRAL_PWM_)));
        }
        mc.channels[STEER_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        break;
      }

      case DriveState::SLOWING: {
        // No fixed creep constant any more — nav2 is already scaling the
        // command down, so the velocity map handles this on its own.
        const double target = (lin > 0.0) ? throttle_pwm_for(lin) : NEUTRAL_PWM_;
        mc.channels[THROTTLE_CH_] =
          static_cast<uint16_t>(std::lround(slew_throttle(target)));
        mc.channels[STEER_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        break;
      }

      case DriveState::STOPPING:
        // INSTANT snap to neutral, ramp deliberately BYPASSED. This is a
        // pre-pivot settle whose whole purpose is to get the chassis
        // physically stopped inside STOP_DURATION_MS_ — ramping here would
        // eat most of that window and the robot would still be rolling when
        // PIVOT began. STOP_DURATION_MS_ is unchanged at 2000.
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
        reset_slew();
        break;

      case DriveState::PIVOT: {
        // Throttle stays hard neutral; steer is not ramped by design.
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        reset_slew();

        // last_ang_ > 0 => LEFT (ASSUMPTION, verify on bench)
        const bool turning_left = (last_ang_ > 0.0);

        if (!pivot_check_set_) {
          current_pivot_output_ = turning_left ? PIVOT_LEFT_BASE_ : PIVOT_RIGHT_BASE_;
          pivot_check_start_    = now();
          pivot_start_yaw_      = odom_yaw_;
          pivot_check_set_      = true;
          RCLCPP_INFO(get_logger(), "PIVOT: direction=%s base_output=%.0f",
            turning_left ? "LEFT" : "RIGHT", current_pivot_output_);
        } else if (odom_ok_) {
          const auto window_ms =
            (now() - pivot_check_start_).nanoseconds() / 1'000'000;

          if (window_ms >= PIVOT_CHECK_MS_) {
            double yaw_delta = std::fabs(odom_yaw_ - pivot_start_yaw_);
            if (yaw_delta > M_PI) yaw_delta = 2.0 * M_PI - yaw_delta;

            if (yaw_delta < PIVOT_ROT_THRESHOLD_) {
              RCLCPP_WARN(get_logger(),
                "PIVOT: insufficient rotation (yaw_delta=%.3f rad in %ldms), "
                "output stays at %.0f (ramping disabled)",
                yaw_delta, static_cast<long>(window_ms), current_pivot_output_);
            } else {
              RCLCPP_INFO(get_logger(),
                "PIVOT: rotating OK (yaw_delta=%.3f rad), output stays at %.0f",
                yaw_delta, current_pivot_output_);
            }

            pivot_check_start_ = now();
            pivot_start_yaw_   = odom_yaw_;
          }
        }

        mc.channels[STEER_CH_] = static_cast<uint16_t>(
          std::clamp(current_pivot_output_, PIVOT_MIN_, PIVOT_MAX_));
        break;
      }

      case DriveState::STALLED:
        break;
    }

    RCLCPP_INFO(get_logger(),
      "cmd_vel [lin=%.3f, ang=%.3f] -> state=%d target=%.0f slew=%.0f "
      "[throttle=%d, steer=%d]",
      msg->linear.x, msg->angular.z, static_cast<int>(state_),
      throttle_pwm_for(lin), slew_current_,
      mc.channels[THROTTLE_CH_], mc.channels[STEER_CH_]);

    output_is_nonzero_ =
      (mc.channels[THROTTLE_CH_] != static_cast<uint16_t>(NEUTRAL_PWM_) ||
       mc.channels[STEER_CH_]    != static_cast<uint16_t>(NEUTRAL_PWM_));

    // ── Stall detection (full stop — separate from the slew limiter) ─────
    // The releasing phase returns early above, so it can never land here.
    const bool in_moving_state =
      (state_ == DriveState::STRAIGHT ||
       state_ == DriveState::SLOWING  ||
       state_ == DriveState::PIVOT);

    if (in_moving_state && output_is_nonzero_ && odom_ok_) {
      if (!stall_window_set_) {
        stall_window_start_ = now();
        stall_window_x_     = odom_x_;
        stall_window_y_     = odom_y_;
        stall_window_set_   = true;
      } else {
        const auto window_ms =
          (now() - stall_window_start_).nanoseconds() / 1'000'000;

        if (window_ms >= STALL_CHECK_MS_) {
          const double dx   = odom_x_ - stall_window_x_;
          const double dy   = odom_y_ - stall_window_y_;
          const double dist = std::sqrt(dx * dx + dy * dy);

          if (dist < STALL_DIST_THRESHOLD_) {
            RCLCPP_WARN(get_logger(),
              "STALL DETECTED: moved only %.4fm in %ldms while commanding motion. "
              "Zeroing output for %dms (INSTANT, ramp bypassed).",
              dist, static_cast<long>(window_ms), STALL_ZERO_MS_);
            state_            = DriveState::STALLED;
            stall_zero_start_ = now();
            mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
            mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
            reset_slew();
            reset_stall_window();
            reset_pivot_state();
            reset_straight_state();
          } else {
            reset_stall_window();
          }
        }
      }
    } else {
      reset_stall_window();
    }

    publish_override(mc, "drive");
  }

  double apply_deadzone(double v) const {
    return (std::fabs(v) < DEADZONE_) ? 0.0 : v;
  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr    sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr      state_sub_;
  rclcpp::Subscription<mavros_msgs::msg::RCIn>::SharedPtr       rc_in_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr polygon_marker_pub_;

  rclcpp::Publisher<mavros_msgs::msg::OverrideRCIn>::SharedPtr  pub_;
  rclcpp::TimerBase::SharedPtr                                  control_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrovBridgeDrive>());
  rclcpp::shutdown();
  return 0;
}