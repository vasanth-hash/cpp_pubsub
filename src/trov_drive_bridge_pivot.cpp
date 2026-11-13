// trov_drive_bridge_pivot.cpp   (2026-11-11)
//
// Nav2 cmd_vel -> ArduPilot Rover RC override, PIVOT-ONLY skid-steer bridge.
// Replaces the ARC behaviour of trov_drive_bridge_outdoor.cpp (that file is
// unchanged and still builds as `drive_outdoor`).
//
// Motion model — the robot either drives (near-)straight or turns in place:
//   IDLE        no fresh cmd / disarmed            -> release all channels (RC passes through)
//   HOLD        Nav2 says stop                     -> neutral, immediately
//   DRIVE       forward, closed-loop speed (FAST-LIO) + small heading trim from the
//               yaw-rate loop. Steering authority is capped (drive_steer_trim_max),
//               so it can never become a big arc.
//   PIVOT_STOP  a turn is needed: neutral until the robot has stopped (or brake_time_s)
//               — also clears the ESC brake-first lockout
//   PIVOT       in-place rotation, CLOSED LOOP on yaw rate:
//                 dev = feedforward + Kp*err + Ki*∫err
//                 stiction: no rotation for pivot_stuck_time_s -> ramp dev at
//                   pivot_breakaway_rate PWM/s; the moment it moves -> drop to the
//                   kinetic level (breakaway dev * pivot_kinetic_ratio)
//                 overspeed: rate > max(pivot_over_rate_abs, ratio*target) on ANY
//                   tick -> dev to pivot_min_dev; rate > pivot_hard_max_rate -> neutral
//                 the breakaway level is remembered and used to start the next pivot
//   SETTLE      after a pivot: neutral until rotation has stopped (pivot_settle_s max)
//
// When to pivot: from Nav2's intent. The raw controller output (/cmd_vel) is used if
// fresh, so a "rotate" or "stop" request acts at once instead of waiting for the
// velocity smoother to ramp v down (that caused ~3 m overshoot). Pivot when
//   |w| > pivot_w_enter  and  turn radius |v/w| < pivot_radius_enter
// and leave when |w| < pivot_w_exit or radius > pivot_radius_exit (hysteresis).
// 2026-11-12: when Nav2's last raw command was a STOP and nothing newer has come
// (goal reached: controller sends one zero twist, then goes silent), keep obeying
// that stop instead of falling back to the smoothed command, which ramps down at
// the velocity smoother's max_decel (0.1 m/s^2) and made the robot creep/stall/creep
// for ~6 s (~1.7 m) after every goal. Param hold_on_nav2_stop (default true).
//
// Yaw-rate measurement (yaw_rate_source):
//   fused (default): lidar-IMU gyro rotated into base_link (200 Hz, no lag) minus a
//          bias that is continuously trimmed against FAST-LIO (/twist) whenever the
//          robot is straight/stationary. Tested 2026-11-11 on bag loctest_153446:
//          gyro scale vs FAST-LIO 1.012, FAST-LIO lags gyro ~0.03 s (+~0.1 s EMA in /twist).
//          NOTE: /rslidar_imu_data says frame "lidar_link" but its axes are imu_link
//          (lidar_link->imu_link static TF from the FAST-LIO launch) — imu_frame param.
//   imu | twist | odom : single source. Fallback order when stale: imu -> twist -> odom.
//
// Collision self-check (same as drive_outdoor): /scan points inside collision_polygon
// -> neutral, latched collision_stop_hold_ms; stale scan / TF failure -> neutral.
//
// RC enable gate (2026-11-11): the bridge only drives while RC channel
// rc_enable_channel (default 8, /mavros/rc/in) is > rc_enable_pwm (default 1900).
// Otherwise it does nothing: on the transition it sends release (0) for
// rc_release_s so the FC drops any held override, then publishes NOTHING.
// No / stale /mavros/rc/in (rc_in_timeout_s) counts as disabled.
// Set rc_enable_channel:=0 to turn the gate off (old behaviour).
//
// Debug: /drive_bridge/status (std_msgs/String, JSON, 10 Hz).
// Nearly all params are live-tunable: ros2 param set /cmd_vel_rc_bridge <name> <val>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <mavros_msgs/msg/override_rc_in.hpp>
#include <mavros_msgs/msg/rc_in.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr uint16_t RC_RELEASE = 0;
constexpr double   EPS = 1e-6;
constexpr double   INF = std::numeric_limits<double>::infinity();
double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
double sgn(double v) { return v >= 0.0 ? 1.0 : -1.0; }
double ang_diff(double a, double b) { return std::atan2(std::sin(a - b), std::cos(a - b)); }
}  // namespace

class PivotBridge : public rclcpp::Node {
public:
  PivotBridge() : Node("cmd_vel_rc_bridge") {
    // ---- topics ----
    input_topic_  = declare_parameter<std::string>("input_topic", "/cmd_vel_smoothed");
    intent_topic_ = declare_parameter<std::string>("intent_topic", "/cmd_vel");   // "" = off
    output_topic_ = declare_parameter<std::string>("output_topic", "/mavros/rc/override");
    imu_topic_    = declare_parameter<std::string>("imu_topic", "/rslidar_imu_data");
    imu_frame_    = declare_parameter<std::string>("imu_frame", "imu_link");
    twist_topic_  = declare_parameter<std::string>("twist_topic", "/twist");
    odom_topic_   = declare_parameter<std::string>("odom_topic", "/odom");
    rate_hz_      = declare_parameter<double>("publish_rate_hz", 50.0);
    watchdog_s_   = declare_parameter<double>("watchdog_s", 0.3);
    intent_timeout_s_ = declare_parameter<double>("intent_timeout_s", 0.3);
    hold_on_nav2_stop_ = declare_parameter<bool>("hold_on_nav2_stop", true);

    // ---- channels / PWM ----
    steer_ch_ = declare_parameter<int>("steer_channel", 1);
    thr_ch_   = declare_parameter<int>("throttle_channel", 3);
    arm_ch_   = declare_parameter<int>("arm_channel", 6);
    neutral_  = declare_parameter<double>("pwm_neutral", 1501.0);
    pwm_min_  = declare_parameter<double>("pwm_min", 1200.0);
    pwm_max_  = declare_parameter<double>("pwm_max", 1800.0);
    steer_sign_ = declare_parameter<int>("steer_sign", -1) >= 0 ? 1.0 : -1.0;  // w>0 (left) -> below neutral
    require_armed_ = declare_parameter<bool>("require_armed", true);
    rc_in_topic_     = declare_parameter<std::string>("rc_in_topic", "/mavros/rc/in");
    rc_enable_ch_    = declare_parameter<int>("rc_enable_channel", 8);       // 1-based, 0 = gate off
    rc_enable_pwm_   = declare_parameter<double>("rc_enable_pwm", 1900.0);   // enabled when > this
    rc_in_timeout_s_ = declare_parameter<double>("rc_in_timeout_s", 0.5);
    rc_release_s_    = declare_parameter<double>("rc_release_s", 0.5);

    // ---- yaw-rate measurement ----
    yaw_src_         = declare_parameter<std::string>("yaw_rate_source", "fused");
    gyro_lpf_hz_     = declare_parameter<double>("gyro_lpf_hz", 15.0);
    bias_tau_s_      = declare_parameter<double>("gyro_bias_tau_s", 5.0);
    imu_timeout_s_   = declare_parameter<double>("imu_timeout_s", 0.1);
    twist_timeout_s_ = declare_parameter<double>("twist_timeout_s", 0.3);
    odom_timeout_s_  = declare_parameter<double>("odom_timeout_s", 0.3);

    // ---- when to pivot ----
    v_eps_          = declare_parameter<double>("v_eps", 0.03);
    w_eps_          = declare_parameter<double>("w_eps", 0.05);
    pivot_w_enter_  = declare_parameter<double>("pivot_w_enter", 0.12);
    pivot_w_exit_   = declare_parameter<double>("pivot_w_exit", 0.06);
    pivot_r_enter_  = declare_parameter<double>("pivot_radius_enter", 3.0);
    pivot_r_exit_   = declare_parameter<double>("pivot_radius_exit", 6.0);
    pivot_min_time_ = declare_parameter<double>("pivot_min_time_s", 0.5);

    // ---- pivot closed loop ----
    max_pivot_rate_   = declare_parameter<double>("max_pivot_rate", 0.3);
    min_pivot_rate_   = declare_parameter<double>("min_pivot_rate", 0.12);
    pivot_ff_         = declare_parameter<double>("pivot_steer_pwm", 100.0);   // start dev
    pivot_kp_         = declare_parameter<double>("pivot_kp", 150.0);          // PWM per rad/s
    pivot_ki_         = declare_parameter<double>("pivot_ki", 200.0);          // PWM per rad
    pivot_min_dev_    = declare_parameter<double>("pivot_min_dev", 60.0);
    pivot_max_dev_    = declare_parameter<double>("pivot_max_dev", 260.0);
    stuck_rate_       = declare_parameter<double>("pivot_stuck_rate", 0.04);
    stuck_time_s_     = declare_parameter<double>("pivot_stuck_time_s", 0.3);
    breakaway_rate_   = declare_parameter<double>("pivot_breakaway_rate", 120.0); // PWM/s
    kinetic_ratio_    = declare_parameter<double>("pivot_kinetic_ratio", 0.6);
    learn_breakaway_  = declare_parameter<bool>("pivot_learn_breakaway", true);
    over_rate_ratio_  = declare_parameter<double>("pivot_over_rate_ratio", 1.5);
    over_rate_abs_    = declare_parameter<double>("pivot_over_rate_abs", 0.2);
    hard_max_rate_    = declare_parameter<double>("pivot_hard_max_rate", 0.8);
    brake_time_s_     = declare_parameter<double>("brake_time_s", 1.0);
    stop_speed_       = declare_parameter<double>("pivot_stop_speed", 0.05);
    settle_s_         = declare_parameter<double>("pivot_settle_s", 0.6);
    settle_rate_      = declare_parameter<double>("pivot_settle_rate", 0.05);

    // ---- drive closed loop ----
    drive_cl_         = declare_parameter<bool>("drive_speed_closed_loop", true);
    drive_fwd_pwm_    = declare_parameter<double>("drive_fwd_thr_pwm", 1540.0);  // ~0.6 m/s open loop
    drive_ref_speed_  = declare_parameter<double>("drive_ref_speed", 0.6);      // speed at drive_fwd_thr_pwm
    drive_start_pwm_  = declare_parameter<double>("drive_start_pwm", 1515.0);   // ~deadband edge
    drive_kp_         = declare_parameter<double>("drive_kp", 30.0);            // PWM per m/s
    drive_ki_         = declare_parameter<double>("drive_ki", 20.0);            // PWM per m
    drive_thr_max_    = declare_parameter<double>("drive_thr_max_pwm", 1575.0);
    max_fwd_speed_    = declare_parameter<double>("max_fwd_speed", 0.65);
    steer_trim_kp_    = declare_parameter<double>("drive_steer_kp", 60.0);      // PWM per rad/s
    steer_trim_max_   = declare_parameter<double>("drive_steer_trim_max", 30.0);

    // ---- collision self-check ----
    collision_polygon_flat_ = declare_parameter<std::vector<double>>(
      "collision_polygon", {1.0, 0.5, 1.3, 0.5, 1.3, -0.5, 1.0, -0.5});
    collision_min_points_    = declare_parameter<int>("collision_min_points", 5);
    collision_base_frame_    = declare_parameter<std::string>("collision_base_frame", "base_link");
    scan_topic_              = declare_parameter<std::string>("scan_topic", "/scan");
    collision_check_enabled_ = declare_parameter<bool>("collision_check_enabled", true);
    collision_hold_s_        = declare_parameter<double>("collision_stop_hold_ms", 3000.0) / 1000.0;
    scan_timeout_s_          = declare_parameter<double>("collision_scan_timeout_s", 1.0);
    if (collision_polygon_flat_.size() < 6 || collision_polygon_flat_.size() % 2 != 0) {
      RCLCPP_ERROR(get_logger(), "collision_polygon malformed — using default forward box.");
      collision_polygon_flat_ = {1.0, 0.5, 1.2, 0.5, 1.2, -0.5, 1.0, -0.5};
    }
    for (size_t i = 0; i + 1 < collision_polygon_flat_.size(); i += 2)
      collision_polygon_.push_back({collision_polygon_flat_[i], collision_polygon_flat_[i + 1]});

    steer_idx_ = steer_ch_ - 1; thr_idx_ = thr_ch_ - 1; arm_idx_ = arm_ch_ - 1;
    rc_gate_t_ = now();   // same clock type as tick() (avoid time-source mismatch throw)

    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    pub_ = create_publisher<mavros_msgs::msg::OverrideRCIn>(output_topic_, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/drive_bridge/status", 10);
    polygon_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/collision_check_polygon", 10);

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(input_topic_, 10,
      [this](geometry_msgs::msg::Twist::SharedPtr m) {
        std::lock_guard<std::mutex> lk(mtx_);
        cmd_v_ = m->linear.x; cmd_w_ = m->angular.z; cmd_t_ = now(); have_cmd_ = true; });
    if (!intent_topic_.empty())
      intent_sub_ = create_subscription<geometry_msgs::msg::Twist>(intent_topic_, 10,
        [this](geometry_msgs::msg::Twist::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mtx_);
          int_v_ = m->linear.x; int_w_ = m->angular.z; int_t_ = now(); have_int_ = true; });
    rc_in_sub_ = create_subscription<mavros_msgs::msg::RCIn>(rc_in_topic_, rclcpp::SensorDataQoS(),
      [this](mavros_msgs::msg::RCIn::SharedPtr m) {
        std::lock_guard<std::mutex> lk(mtx_);
        const int i = rc_enable_ch_ - 1;
        rc_enable_val_ = (i >= 0 && static_cast<size_t>(i) < m->channels.size()) ? m->channels[i] : 0;
        rc_in_t_ = now(); have_rc_in_ = true; });
    state_sub_ = create_subscription<mavros_msgs::msg::State>("/mavros/state", 10,
      std::bind(&PivotBridge::on_state, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PivotBridge::on_imu, this, std::placeholders::_1));
    twist_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      twist_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PivotBridge::on_twist, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PivotBridge::on_odom, this, std::placeholders::_1));
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(scan_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PivotBridge::on_scan, this, std::placeholders::_1));
    collision_enable_sub_ = create_subscription<std_msgs::msg::Bool>("/collision_check_enable", 10,
      [this](std_msgs::msg::Bool::SharedPtr m) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (m->data != collision_check_enabled_)
          RCLCPP_WARN(get_logger(), "COLLISION CHECK enforcement -> %s", m->data ? "ENABLED" : "DISABLED");
        collision_check_enabled_ = m->data; });

    param_cb_ = add_on_set_parameters_callback(
      std::bind(&PivotBridge::on_params, this, std::placeholders::_1));

    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / std::max(5.0, rate_hz_))),
      std::bind(&PivotBridge::tick, this));

    std::ostringstream o;
    for (const auto & pt : collision_polygon_) o << "(" << pt.x << "," << pt.y << ") ";
    RCLCPP_WARN(get_logger(), "=== PIVOT-ONLY bridge BUILD %s %s ===", __DATE__, __TIME__);
    RCLCPP_WARN(get_logger(),
      "in=%s intent=%s out=%s @%.0fHz | yaw_rate_source=%s imu=%s(frame %s) twist=%s odom=%s | "
      "pivot: rate %.2f-%.2f rad/s, enter |w|>%.2f & r<%.1fm, max_dev %.0f | drive: %s, fwd %.0f@%.2fm/s, trim<=%.0f | "
      "collision: %s min_points=%d polygon=[%s]",
      input_topic_.c_str(), intent_topic_.empty() ? "(off)" : intent_topic_.c_str(), output_topic_.c_str(),
      rate_hz_, yaw_src_.c_str(), imu_topic_.c_str(), imu_frame_.c_str(), twist_topic_.c_str(),
      odom_topic_.c_str(), min_pivot_rate_, max_pivot_rate_, pivot_w_enter_, pivot_r_enter_,
      pivot_max_dev_, drive_cl_ ? "closed-loop speed" : "fixed PWM", drive_fwd_pwm_, drive_ref_speed_,
      steer_trim_max_, collision_check_enabled_ ? "ON" : "OFF", collision_min_points_, o.str().c_str());
  }

private:
  enum class St { IDLE, HOLD, DRIVE, PIVOT_STOP, PIVOT, SETTLE };
  struct Pt2 { double x; double y; };
  static const char * name(St s) {
    switch (s) {
      case St::IDLE: return "IDLE"; case St::HOLD: return "HOLD"; case St::DRIVE: return "DRIVE";
      case St::PIVOT_STOP: return "PIVOT_STOP"; case St::PIVOT: return "PIVOT"; case St::SETTLE: return "SETTLE";
    }
    return "?";
  }

  // ================= sensors =================
  void on_state(const mavros_msgs::msg::State::SharedPtr m) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (armed_ && !m->armed) { RCLCPP_WARN(get_logger(), "DISARM detected — IDLE."); set_state(St::IDLE, now()); }
    armed_ = m->armed;
  }

  bool lookup_imu_rotation() {
    if (have_R_) return true;
    try {
      const auto tf = tf_buffer_->lookupTransform("base_link", imu_frame_, tf2::TimePointZero);
      tf2::Quaternion q(tf.transform.rotation.x, tf.transform.rotation.y,
                        tf.transform.rotation.z, tf.transform.rotation.w);
      tf2::Matrix3x3 m(q);
      for (int i = 0; i < 3; ++i) R_z_[i] = m[2][i];   // only the z row is needed (yaw rate)
      have_R_ = true;
      RCLCPP_INFO(get_logger(), "IMU %s -> base_link rotation resolved (z row %.3f %.3f %.3f)",
        imu_frame_.c_str(), R_z_[0], R_z_[1], R_z_[2]);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "waiting for TF base_link <- %s (%s); yaw rate falls back to twist/odom", imu_frame_.c_str(), ex.what());
    }
    return have_R_;
  }

  void on_imu(const sensor_msgs::msg::Imu::SharedPtr m) {
    if (!lookup_imu_rotation()) return;
    const double wz = R_z_[0] * m->angular_velocity.x + R_z_[1] * m->angular_velocity.y +
                      R_z_[2] * m->angular_velocity.z;
    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();
    const double dt = have_imu_ ? clampd((t - imu_t_).seconds(), 1e-4, 0.1) : 0.005;
    const double a = 1.0 - std::exp(-2.0 * M_PI * gyro_lpf_hz_ * dt);
    gyro_f_ = have_imu_ ? gyro_f_ + a * (wz - gyro_f_) : wz;
    // slow average of the gyro for bias trimming (matches /twist's smoothing roughly)
    const double a2 = 1.0 - std::exp(-dt / 0.15);
    gyro_slow_ = have_imu_ ? gyro_slow_ + a2 * (wz - gyro_slow_) : wz;
    imu_t_ = t; have_imu_ = true;
  }

  void on_twist(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr m) {
    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();
    const double dt = have_twist_ ? clampd((t - twist_t_).seconds(), 1e-3, 0.5) : 0.1;
    twist_wz_ = m->twist.twist.angular.z;
    twist_vx_ = m->twist.twist.linear.x;
    twist_t_ = t; have_twist_ = true;
    // bias trim: only when not turning (bias is observable, lag does not matter)
    if (have_imu_ && (t - imu_t_).seconds() < imu_timeout_s_ &&
        std::fabs(twist_wz_) < 0.05 && std::fabs(gyro_slow_) < 0.15) {
      const double k = 1.0 - std::exp(-dt / std::max(0.5, bias_tau_s_));
      gyro_bias_ += k * ((gyro_slow_ - twist_wz_) - gyro_bias_);
      bias_ok_ = true;
    }
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr m) {
    const auto & q = m->pose.pose.orientation;
    const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();
    if (have_odom_) {
      const double dt = (t - odom_t_).seconds();
      if (dt > 1e-3) {
        odom_wz_ = 0.4 * ang_diff(yaw, odom_yaw_) / dt + 0.6 * odom_wz_;
        const double dx = m->pose.pose.position.x - odom_x_, dy = m->pose.pose.position.y - odom_y_;
        // signed forward speed along the robot heading
        odom_vx_ = 0.3 * ((dx * std::cos(yaw) + dy * std::sin(yaw)) / dt) + 0.7 * odom_vx_;
      }
    }
    odom_yaw_ = yaw; odom_x_ = m->pose.pose.position.x; odom_y_ = m->pose.pose.position.y;
    odom_t_ = t; have_odom_ = true;
  }

  // caller holds mtx_. Returns yaw rate (rad/s, CCW+) and which source was used.
  double yaw_rate(const rclcpp::Time & t, const char ** src) const {
    const bool imu_ok   = have_imu_ && (t - imu_t_).seconds() < imu_timeout_s_;
    const bool twist_ok = have_twist_ && (t - twist_t_).seconds() < twist_timeout_s_;
    const bool odom_ok  = have_odom_ && (t - odom_t_).seconds() < odom_timeout_s_;
    if ((yaw_src_ == "fused" || yaw_src_ == "imu") && imu_ok) {
      *src = (yaw_src_ == "fused") ? (bias_ok_ ? "fused" : "imu(bias pending)") : "imu";
      return gyro_f_ - (yaw_src_ == "fused" ? gyro_bias_ : 0.0);
    }
    if (yaw_src_ != "odom" && twist_ok) { *src = "twist"; return twist_wz_; }
    if (odom_ok) { *src = "odom"; return odom_wz_; }
    *src = "none"; return 0.0;
  }
  // caller holds mtx_
  double fwd_speed(const rclcpp::Time & t, bool * ok) const {
    if (have_twist_ && (t - twist_t_).seconds() < twist_timeout_s_) { *ok = true; return twist_vx_; }
    if (have_odom_ && (t - odom_t_).seconds() < odom_timeout_s_) { *ok = true; return odom_vx_; }
    *ok = false; return 0.0;
  }

  // ================= collision =================
  bool point_in_polygon(double x, double y) const {
    bool inside = false;
    const size_t n = collision_polygon_.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
      const double xi = collision_polygon_[i].x, yi = collision_polygon_[i].y;
      const double xj = collision_polygon_[j].x, yj = collision_polygon_[j].y;
      if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) inside = !inside;
    }
    return inside;
  }
  void set_collision_stop(bool stop, int pts, const char * why, const rclcpp::Time & t) {  // holds mtx_
    if (stop != collision_stop_) {
      RCLCPP_WARN(get_logger(), "COLLISION CHECK: -> %s (reason=%s, points_inside=%d, min_points=%d)",
        stop ? "STOP" : "CLEAR", why, pts, collision_min_points_);
      if (stop) { collision_hold_until_ = t + rclcpp::Duration::from_seconds(collision_hold_s_); collision_hold_set_ = true; }
    }
    collision_stop_ = stop; last_points_inside_ = pts;
  }
  bool collision_enforced(const rclcpp::Time & t) const {
    return collision_stop_ || (collision_hold_set_ && t < collision_hold_until_);
  }
  void on_scan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    double tx = 0, ty = 0, cy = 1, sy = 0; bool tf_ok = true;
    if (msg->header.frame_id != collision_base_frame_) {
      try {
        const auto tf = tf_buffer_->lookupTransform(collision_base_frame_, msg->header.frame_id,
          msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
        tx = tf.transform.translation.x; ty = tf.transform.translation.y;
        tf2::Quaternion q(tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w);
        double r, p, yw; tf2::Matrix3x3(q).getRPY(r, p, yw); cy = std::cos(yw); sy = std::sin(yw);
      } catch (const tf2::TransformException &) { tf_ok = false; }
    }
    int pts = 0;
    if (tf_ok) {
      for (size_t i = 0; i < msg->ranges.size(); ++i) {
        const float rg = msg->ranges[i];
        if (!std::isfinite(rg) || rg < msg->range_min || rg > msg->range_max) continue;
        const double a = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
        const double lx = rg * std::cos(a), ly = rg * std::sin(a);
        if (point_in_polygon(tx + cy * lx - sy * ly, ty + sy * lx + cy * ly) && ++pts >= collision_min_points_) break;
      }
    }
    {
      std::lock_guard<std::mutex> lk(mtx_);
      const rclcpp::Time t = now();
      scan_ok_ = true; scan_t_ = t;
      if (!tf_ok) set_collision_stop(true, -1, "tf_lookup_failed", t);
      else set_collision_stop(pts >= collision_min_points_, pts, "scan_check", t);
    }
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = collision_base_frame_; mk.header.stamp = msg->header.stamp;
    mk.ns = "collision_self_check"; mk.id = 0; mk.type = visualization_msgs::msg::Marker::LINE_STRIP;
    mk.action = visualization_msgs::msg::Marker::ADD; mk.scale.x = 0.05;
    mk.lifetime = rclcpp::Duration::from_seconds(0.5); mk.color.b = 1.0f; mk.color.a = 1.0f;
    for (const auto & pt : collision_polygon_) { geometry_msgs::msg::Point p; p.x = pt.x; p.y = pt.y; mk.points.push_back(p); }
    if (!mk.points.empty()) mk.points.push_back(mk.points.front());
    polygon_marker_pub_->publish(mk);
  }

  // ================= output helpers =================
  void publish_raw(double steer_pwm, double thr_pwm, bool release) {
    mavros_msgs::msg::OverrideRCIn mc;
    mc.channels.fill(RC_RELEASE);
    if (!release) {
      mc.channels[steer_idx_] = static_cast<uint16_t>(std::lround(clampd(steer_pwm, pwm_min_, pwm_max_)));
      mc.channels[thr_idx_]   = static_cast<uint16_t>(std::lround(clampd(thr_pwm, pwm_min_, pwm_max_)));
    }
    if (arm_idx_ >= 0 && static_cast<size_t>(arm_idx_) < mc.channels.size()) mc.channels[arm_idx_] = RC_RELEASE;
    pub_->publish(mc);
    out_steer_ = release ? 0 : steer_pwm; out_thr_ = release ? 0 : thr_pwm;
  }
  void neutral() { publish_raw(neutral_, neutral_, false); }
  void release() { publish_raw(0, 0, true); }

  void set_state(St s, const rclcpp::Time & t) {   // holds mtx_
    if (s == state_) return;
    RCLCPP_INFO(get_logger(), "state %s -> %s", name(state_), name(s));
    state_ = s; state_t_ = t;
    if (s == St::PIVOT) {
      stuck_since_set_ = false; breakaway_ = false; pivot_int_ = 0.0;
      pivot_dev_ = clampd(learn_breakaway_ && learned_dev_ > 0 ? std::max(pivot_ff_, 0.8 * learned_dev_) : pivot_ff_,
                          pivot_min_dev_, pivot_max_dev_);
    }
    if (s == St::DRIVE) drive_int_ = 0.0;
  }

  // ================= main loop =================
  void tick() {
    std::lock_guard<std::mutex> lk(mtx_);
    const rclcpp::Time t = now();
    const double dt = have_tick_ ? clampd((t - last_tick_).seconds(), 1e-3, 0.2) : 1.0 / rate_hz_;
    last_tick_ = t; have_tick_ = true;
    const char * src = "none";
    const double wz = yaw_rate(t, &src);
    bool spd_ok = false; const double spd = fwd_speed(t, &spd_ok);

    // ---- gates ----
    if (rc_enable_ch_ > 0) {
      const bool rc_fresh = have_rc_in_ && (t - rc_in_t_).seconds() < rc_in_timeout_s_;
      const bool en = rc_fresh && rc_enable_val_ > rc_enable_pwm_;
      if (en != rc_enabled_) {
        RCLCPP_WARN(get_logger(), "RC ch%d gate -> %s (pwm %d%s)", rc_enable_ch_, en ? "ENABLED" : "DISABLED",
          rc_enable_val_, rc_fresh ? "" : ", rc/in stale");
        rc_enabled_ = en; rc_gate_t_ = t;
      }
      if (!en) {
        set_state(St::IDLE, t);
        if ((t - rc_gate_t_).seconds() < rc_release_s_) release();   // let go of the FC, then stay silent
        else { out_steer_ = 0; out_thr_ = 0; }
        return status(t, wz, src, spd);
      }
    }
    if (require_armed_ && !armed_) { set_state(St::IDLE, t); release(); return status(t, wz, src, spd); }
    const double since = have_cmd_ ? (t - cmd_t_).seconds() : INF;
    if (since > watchdog_s_) {
      if (state_ != St::IDLE) RCLCPP_INFO(get_logger(), "no cmd_vel -> releasing to remote (IDLE)");
      set_state(St::IDLE, t); release(); return status(t, wz, src, spd);
    }
    if (collision_check_enabled_) {
      const double sa = scan_ok_ ? (t - scan_t_).seconds() : INF;
      if (sa > scan_timeout_s_ || collision_enforced(t)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s — holding NEUTRAL",
          sa > scan_timeout_s_ ? "COLLISION CHECK: no fresh scan" : "COLLISION STOP active");
        set_state(St::HOLD, t); neutral(); return status(t, wz, src, spd);
      }
    }

    // ---- what does Nav2 want? raw controller output if fresh, else smoothed ----
    const bool int_ok = have_int_ && (t - int_t_).seconds() < intent_timeout_s_;
    // Nav2 went quiet after commanding a stop -> stay stopped (ignore the smoother's ramp)
    const bool nav2_stopped = hold_on_nav2_stop_ && have_int_ && !int_ok &&
      std::fabs(int_v_) < v_eps_ && std::fabs(int_w_) < w_eps_;
    if (nav2_stopped != nav2_stopped_) {
      if (nav2_stopped) RCLCPP_INFO(get_logger(), "Nav2 stopped (last raw cmd was zero) -> holding, ignoring smoother ramp (smoothed v %.2f)", cmd_v_);
      nav2_stopped_ = nav2_stopped;
    }
    const double iv = int_ok ? int_v_ : (nav2_stopped ? 0.0 : cmd_v_);
    const double iw = int_ok ? int_w_ : (nav2_stopped ? 0.0 : cmd_w_);
    const bool in_pivot = (state_ == St::PIVOT_STOP || state_ == St::PIVOT);
    const double radius = std::fabs(iw) > EPS ? std::fabs(iv / iw) : INF;
    const bool want_pivot = in_pivot
      ? !(std::fabs(iw) < pivot_w_exit_ || radius > pivot_r_exit_)
      : (std::fabs(iw) > pivot_w_enter_ && radius < pivot_r_enter_);
    const bool want_stop = std::fabs(iv) < v_eps_ && std::fabs(iw) < w_eps_;

    // ---- PIVOT family ----
    if (want_pivot || (state_ == St::PIVOT && (t - state_t_).seconds() < pivot_min_time_ && !want_stop)) {
      const double dir = sgn(iw);                      // +1 = CCW (left)
      if (!in_pivot) { pivot_dir_ = dir; set_state(St::PIVOT_STOP, t); }
      else if (want_pivot && dir != pivot_dir_) {      // reversal -> stop first
        RCLCPP_INFO(get_logger(), "PIVOT direction reversed — stopping first");
        pivot_dir_ = dir; set_state(St::PIVOT_STOP, t);
      }
      if (state_ == St::PIVOT_STOP) {
        neutral();
        const double el = (t - state_t_).seconds();
        const bool stopped = spd_ok && std::fabs(spd) < stop_speed_ && el > 0.2;
        if (el >= brake_time_s_ || (stopped && el >= 0.5 * brake_time_s_)) set_state(St::PIVOT, t);
        return status(t, wz, src, spd);
      }
      // closed-loop yaw rate
      const double target = clampd(std::fabs(iw), min_pivot_rate_, max_pivot_rate_);
      const double m = pivot_dir_ * wz;                // measured rate in the commanded direction
      const double err = target - m;
      if (std::fabs(wz) > hard_max_rate_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
          "PIVOT hard limit: |rate| %.2f > %.2f -> neutral", std::fabs(wz), hard_max_rate_);
        pivot_dev_ = pivot_min_dev_; pivot_int_ = 0.0; neutral(); return status(t, wz, src, spd);
      }
      if (m > std::max(over_rate_abs_, over_rate_ratio_ * target)) {
        if (pivot_dev_ > pivot_min_dev_)
          RCLCPP_WARN(get_logger(), "PIVOT overspeed: %.2f > %.2f rad/s -> dev %.0f -> %.0f",
            m, std::max(over_rate_abs_, over_rate_ratio_ * target), pivot_dev_, pivot_min_dev_);
        pivot_dev_ = pivot_min_dev_; pivot_int_ = 0.0; breakaway_ = false; stuck_since_set_ = false;
      } else if (m < stuck_rate_) {
        if (!stuck_since_set_) { stuck_since_ = t; stuck_since_set_ = true; }
        if ((t - stuck_since_).seconds() > stuck_time_s_) {
          if (!breakaway_) { breakaway_ = true; RCLCPP_INFO(get_logger(), "PIVOT stuck -> breakaway ramp from %.0f", pivot_dev_); }
          pivot_dev_ = std::min(pivot_dev_ + breakaway_rate_ * dt, pivot_max_dev_);
        } else {
          pivot_int_ = clampd(pivot_int_ + err * dt, -1.0, 1.0);
          pivot_dev_ = clampd(pivot_ff_ + pivot_kp_ * err + pivot_ki_ * pivot_int_, pivot_min_dev_, pivot_max_dev_);
        }
      } else {
        stuck_since_set_ = false;
        if (breakaway_) {                              // just broke free: drop to kinetic level
          breakaway_ = false;
          if (learn_breakaway_) learned_dev_ = 0.7 * learned_dev_ + 0.3 * pivot_dev_;
          const double kin = std::max(pivot_min_dev_, kinetic_ratio_ * pivot_dev_);
          RCLCPP_INFO(get_logger(), "PIVOT broke free at dev %.0f (rate %.2f) -> kinetic %.0f", pivot_dev_, m, kin);
          pivot_dev_ = kin;
          pivot_int_ = pivot_ki_ > EPS ? clampd((kin - pivot_ff_) / pivot_ki_, -1.0, 1.0) : 0.0;
        } else {
          pivot_int_ = clampd(pivot_int_ + err * dt, -1.0, 1.0);
          pivot_dev_ = clampd(pivot_ff_ + pivot_kp_ * err + pivot_ki_ * pivot_int_, pivot_min_dev_, pivot_max_dev_);
        }
      }
      // steer_sign: w>0 (CCW/left) maps to below-neutral PWM on this hardware
      publish_raw(neutral_ + steer_sign_ * pivot_dir_ * pivot_dev_, neutral_, false);
      return status(t, wz, src, spd);
    }

    // ---- leaving a pivot: settle (stop rotating) before anything else ----
    if (in_pivot) set_state(St::SETTLE, t);
    if (state_ == St::SETTLE) {
      const double el = (t - state_t_).seconds();
      if (el < settle_s_ && !(std::fabs(wz) < settle_rate_ && el > 0.15)) { neutral(); return status(t, wz, src, spd); }
      set_state(want_stop ? St::HOLD : St::DRIVE, t);
    }

    // ---- STOP ----
    if (want_stop || iv <= v_eps_) { set_state(St::HOLD, t); neutral(); return status(t, wz, src, spd); }

    // ---- DRIVE: closed-loop speed + capped heading trim ----
    set_state(St::DRIVE, t);
    const double v_target = clampd(std::min(cmd_v_, iv), 0.0, max_fwd_speed_);  // smoothed for accel, intent for decel
    double thr;
    if (drive_cl_ && spd_ok) {
      const double ff = v_target > EPS
        ? drive_start_pwm_ + (drive_fwd_pwm_ - drive_start_pwm_) * (v_target / std::max(0.05, drive_ref_speed_))
        : neutral_;
      const double e = v_target - spd;
      drive_int_ = clampd(drive_int_ + e * dt, -1.0, 1.0);
      thr = ff + drive_kp_ * e + drive_ki_ * drive_int_;
    } else {
      thr = v_target > EPS ? drive_fwd_pwm_ : neutral_;
    }
    thr = clampd(thr, neutral_, drive_thr_max_);         // no reverse in DRIVE
    // heading trim: track Nav2's (small) w with the measured yaw rate, capped
    const double trim = clampd(steer_trim_kp_ * (iw - wz) + steer_trim_kp_ * 0.5 * iw, -steer_trim_max_, steer_trim_max_);
    publish_raw(neutral_ + steer_sign_ * trim, thr, false);
    status(t, wz, src, spd);
  }

  void status(const rclcpp::Time & t, double wz, const char * src, double spd) {   // holds mtx_
    if (have_status_ && (t - status_t_).seconds() < 0.1) return;
    status_t_ = t; have_status_ = true;
    char buf[640];
    std::snprintf(buf, sizeof(buf),
      "{\"state\":\"%s\",\"cmd_v\":%.3f,\"cmd_w\":%.3f,\"int_v\":%.3f,\"int_w\":%.3f,\"yaw_rate\":%.3f,"
      "\"src\":\"%s\",\"gyro\":%.3f,\"bias\":%.4f,\"twist_wz\":%.3f,\"odom_wz\":%.3f,\"speed\":%.3f,"
      "\"pivot_dev\":%.0f,\"learned_dev\":%.0f,\"steer\":%.0f,\"thr\":%.0f,\"collision\":%s,\"armed\":%s,"
      "\"rc_gate\":%s,\"rc_ch_pwm\":%d}",
      name(state_), cmd_v_, cmd_w_, int_v_, int_w_, wz, src, gyro_f_, gyro_bias_, twist_wz_, odom_wz_, spd,
      pivot_dev_, learned_dev_, out_steer_, out_thr_, collision_stop_ ? "true" : "false", armed_ ? "true" : "false",
      rc_enable_ch_ <= 0 ? "\"off\"" : (rc_enabled_ ? "true" : "false"), rc_enable_val_);
    std_msgs::msg::String s; s.data = buf; status_pub_->publish(s);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "%s", buf);
  }

  rcl_interfaces::msg::SetParametersResult on_params(const std::vector<rclcpp::Parameter> & ps) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto & p : ps) {
      const auto & n = p.get_name();
      auto D = [&](const char * k, double & v) { if (n == k) v = p.as_double(); };
      auto B = [&](const char * k, bool & v) { if (n == k) v = p.as_bool(); };
      D("watchdog_s", watchdog_s_); D("intent_timeout_s", intent_timeout_s_);
      if (n == "hold_on_nav2_stop") hold_on_nav2_stop_ = p.as_bool();
      D("pwm_neutral", neutral_); D("gyro_lpf_hz", gyro_lpf_hz_); D("gyro_bias_tau_s", bias_tau_s_);
      D("v_eps", v_eps_); D("w_eps", w_eps_); D("pivot_w_enter", pivot_w_enter_); D("pivot_w_exit", pivot_w_exit_);
      D("pivot_radius_enter", pivot_r_enter_); D("pivot_radius_exit", pivot_r_exit_); D("pivot_min_time_s", pivot_min_time_);
      D("max_pivot_rate", max_pivot_rate_); D("min_pivot_rate", min_pivot_rate_); D("pivot_steer_pwm", pivot_ff_);
      D("pivot_kp", pivot_kp_); D("pivot_ki", pivot_ki_); D("pivot_min_dev", pivot_min_dev_); D("pivot_max_dev", pivot_max_dev_);
      D("pivot_stuck_rate", stuck_rate_); D("pivot_stuck_time_s", stuck_time_s_); D("pivot_breakaway_rate", breakaway_rate_);
      D("pivot_kinetic_ratio", kinetic_ratio_); D("pivot_over_rate_ratio", over_rate_ratio_); D("pivot_over_rate_abs", over_rate_abs_);
      D("pivot_hard_max_rate", hard_max_rate_); D("brake_time_s", brake_time_s_); D("pivot_stop_speed", stop_speed_);
      D("pivot_settle_s", settle_s_); D("pivot_settle_rate", settle_rate_);
      D("drive_fwd_thr_pwm", drive_fwd_pwm_); D("drive_ref_speed", drive_ref_speed_); D("drive_start_pwm", drive_start_pwm_);
      D("drive_kp", drive_kp_); D("drive_ki", drive_ki_); D("drive_thr_max_pwm", drive_thr_max_); D("max_fwd_speed", max_fwd_speed_);
      D("drive_steer_kp", steer_trim_kp_); D("drive_steer_trim_max", steer_trim_max_);
      B("drive_speed_closed_loop", drive_cl_); B("pivot_learn_breakaway", learn_breakaway_);
      B("collision_check_enabled", collision_check_enabled_); B("require_armed", require_armed_);
      D("rc_enable_pwm", rc_enable_pwm_); D("rc_in_timeout_s", rc_in_timeout_s_); D("rc_release_s", rc_release_s_);
      if (n == "rc_enable_channel") rc_enable_ch_ = static_cast<int>(p.as_int());
      if (n == "collision_min_points") collision_min_points_ = static_cast<int>(p.as_int());
      if (n == "collision_stop_hold_ms") collision_hold_s_ = p.as_double() / 1000.0;
      if (n == "yaw_rate_source") yaw_src_ = p.as_string();
    }
    rcl_interfaces::msg::SetParametersResult r; r.successful = true; return r;
  }

  // ---- params ----
  std::string input_topic_, intent_topic_, output_topic_, imu_topic_, imu_frame_, twist_topic_, odom_topic_, yaw_src_;
  double rate_hz_{50}, watchdog_s_{0.3}, intent_timeout_s_{0.3};
  bool hold_on_nav2_stop_{true}, nav2_stopped_{false};
  int steer_ch_{1}, thr_ch_{3}, arm_ch_{6}, steer_idx_{0}, thr_idx_{2}, arm_idx_{5};
  double neutral_{1501}, pwm_min_{1200}, pwm_max_{1800}, steer_sign_{-1};
  bool require_armed_{true};
  std::string rc_in_topic_{"/mavros/rc/in"};
  int rc_enable_ch_{8}; double rc_enable_pwm_{1900}, rc_in_timeout_s_{0.5}, rc_release_s_{0.5};
  double gyro_lpf_hz_{15}, bias_tau_s_{5}, imu_timeout_s_{0.1}, twist_timeout_s_{0.3}, odom_timeout_s_{0.3};
  double v_eps_{0.03}, w_eps_{0.05}, pivot_w_enter_{0.12}, pivot_w_exit_{0.06}, pivot_r_enter_{3}, pivot_r_exit_{6}, pivot_min_time_{0.5};
  double max_pivot_rate_{0.3}, min_pivot_rate_{0.12}, pivot_ff_{100}, pivot_kp_{150}, pivot_ki_{200};
  double pivot_min_dev_{60}, pivot_max_dev_{260}, stuck_rate_{0.04}, stuck_time_s_{0.3}, breakaway_rate_{120};
  double kinetic_ratio_{0.6}, over_rate_ratio_{1.5}, over_rate_abs_{0.2}, hard_max_rate_{0.8};
  double brake_time_s_{1.0}, stop_speed_{0.05}, settle_s_{0.6}, settle_rate_{0.05};
  bool learn_breakaway_{true}, drive_cl_{true};
  double drive_fwd_pwm_{1540}, drive_ref_speed_{0.6}, drive_start_pwm_{1515}, drive_kp_{30}, drive_ki_{20};
  double drive_thr_max_{1575}, max_fwd_speed_{0.65}, steer_trim_kp_{60}, steer_trim_max_{30};
  std::vector<double> collision_polygon_flat_; std::vector<Pt2> collision_polygon_;
  int collision_min_points_{9}; std::string collision_base_frame_{"base_link"}, scan_topic_{"/scan"};
  bool collision_check_enabled_{true}; double collision_hold_s_{3}, scan_timeout_s_{1};

  // ---- state ----
  std::mutex mtx_;
  St state_{St::IDLE}; rclcpp::Time state_t_;
  bool armed_{false};
  int rc_enable_val_{0}; bool have_rc_in_{false}, rc_enabled_{false}; rclcpp::Time rc_in_t_, rc_gate_t_;
  double cmd_v_{0}, cmd_w_{0}, int_v_{0}, int_w_{0}; rclcpp::Time cmd_t_, int_t_; bool have_cmd_{false}, have_int_{false};
  bool have_R_{false}; double R_z_[3]{0, 0, 1};
  double gyro_f_{0}, gyro_slow_{0}, gyro_bias_{0}; bool have_imu_{false}, bias_ok_{false}; rclcpp::Time imu_t_;
  double twist_wz_{0}, twist_vx_{0}; bool have_twist_{false}; rclcpp::Time twist_t_;
  double odom_wz_{0}, odom_vx_{0}, odom_yaw_{0}, odom_x_{0}, odom_y_{0}; bool have_odom_{false}; rclcpp::Time odom_t_;
  double pivot_dir_{1}, pivot_dev_{100}, pivot_int_{0}, learned_dev_{0};
  bool breakaway_{false}, stuck_since_set_{false}; rclcpp::Time stuck_since_;
  double drive_int_{0};
  bool have_tick_{false}; rclcpp::Time last_tick_;
  double out_steer_{0}, out_thr_{0};
  bool have_status_{false}; rclcpp::Time status_t_;
  bool collision_stop_{false}, collision_hold_set_{false}, scan_ok_{false}; int last_points_inside_{0};
  rclcpp::Time collision_hold_until_, scan_t_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_; std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<mavros_msgs::msg::OverrideRCIn>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr polygon_marker_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_, intent_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<mavros_msgs::msg::RCIn>::SharedPtr rc_in_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_enable_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PivotBridge>());
  rclcpp::shutdown();
  return 0;
}
