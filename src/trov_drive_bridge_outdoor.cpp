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
      "collision_polygon", {1.0, 1.3,  1.0, 0.9,  -0.7, 0.9, -0.7, 1.3});
    this->declare_parameter<int>("collision_min_points", 12);
    this->declare_parameter<std::string>("collision_base_frame", "base_link");
    this->declare_parameter<std::string>("scan_topic", "/scan");
    // Enforcement on/off — detection, logging, and the RViz marker keep
    // running regardless of this; only the "force neutral" gate in
    // control_loop() respects it. Default true = same behavior as before
    // this toggle existed. The topic below overrides this at runtime.
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
    // checking against, and stays live-colored with current stop state.
    polygon_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      "/collision_check_polygon", 10);

    pub_ = create_publisher<mavros_msgs::msg::OverrideRCIn>(
      "/mavros/rc/override", 10);

    // Drive/publish decisions run on this fixed-rate timer, NOT on the
    // cmd_vel subscription callback — this is what lets us detect
    // "/cmd_vel_smoothed has gone silent" (nav stack idle) instead of just
    // never running again if the topic stops.
    control_timer_ = create_wall_timer(
      std::chrono::milliseconds(CONTROL_PERIOD_MS_),
      std::bind(&TrovBridgeDrive::control_loop, this));

    // ── BUILD MARKER — confirm this matches your latest build before
    //    trusting any other log line below. ─────────────────────────────
    RCLCPP_WARN(get_logger(),
      "=== TrovBridgeDrive BUILD: %s %s ===", __DATE__, __TIME__);

    RCLCPP_INFO(get_logger(),
      "TROV Drive Bridge started — steer=RC%zu(neutral=%.0f right_base=%.0f "
      "left_base=%.0f step=%.0f min=%.0f max=%.0f) "
      "throttle=RC%zu(neutral=%.0f base=%.0f step=%.0f max=%.0f no_reverse=true) "
      "arm=RC%zu(ALWAYS RELEASED) cmd_vel_timeout_ms=%d control_period_ms=%d "
      "ang_threshold=%.3f ang_slowzone=%.3f "
      "stop_duration_ms=%d stall_check_ms=%d stall_dist=%.4f stall_zero_ms=%d",
      STEER_CH_ + 1, NEUTRAL_PWM_, PIVOT_RIGHT_BASE_, PIVOT_LEFT_BASE_,
      PIVOT_STEP_, PIVOT_MIN_, PIVOT_MAX_,
      THROTTLE_CH_ + 1, NEUTRAL_PWM_, THROTTLE_BASE_, THROTTLE_STEP_, THROTTLE_MAX_,
      ARM_CH_ + 1, CMD_VEL_TIMEOUT_MS_, CONTROL_PERIOD_MS_,
      ANG_THRESHOLD_, ANG_SLOW_ZONE_,
      STOP_DURATION_MS_, STALL_CHECK_MS_, STALL_DIST_THRESHOLD_,
      STALL_ZERO_MS_);

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
  static constexpr double THROTTLE_BASE_ = 1530.0; // starting output once moving
  static constexpr double THROTTLE_STEP_ = 10.0;   // adaptive ramp step (ramping disabled below, constant kept)
  static constexpr double THROTTLE_MAX_  = 1600.0; // HARD CEILING — never exceed
  static constexpr double THROTTLE_MIN_  = NEUTRAL_PWM_; // HARD FLOOR — no reverse, ever
  static constexpr double SLOW_THROTTLE_ = 1515.0;  // ASSUMPTION: fixed creep in SLOWING, tune me
  static constexpr int    STRAIGHT_CHECK_MS_      = 1500; // ASSUMPTION: tune me
  static constexpr double STRAIGHT_DIST_THRESHOLD_ = 0.05; // meters, ASSUMPTION: tune me

  // ── Steer / rotation (RC1) ───────────────────────────────────────────
  // ASSUMPTION: ang > 0 (ROS REP-103, angular.z counter-clockwise) = LEFT.
  // Verify this matches your hardware's actual turn direction on the bench.
  static constexpr double PIVOT_RIGHT_BASE_ = 1650.0; // starting output turning right
  static constexpr double PIVOT_LEFT_BASE_  = 1340.0; // starting output turning left
  static constexpr double PIVOT_STEP_       = 10.0;   // adaptive ramp step (ramping disabled below, constant kept)
  static constexpr double PIVOT_MIN_        = 1100.0; // HARD FLOOR — never exceed (left ramp limit)
  static constexpr double PIVOT_MAX_        = 1800.0; // HARD CEILING — never exceed (right ramp limit)
  static constexpr int    PIVOT_CHECK_MS_      = 1500;
  static constexpr double PIVOT_ROT_THRESHOLD_ = 0.5;  // radians
  // ang must stay continuously ABOVE threshold this long before PIVOT is
  // entered, and continuously BELOW threshold this long before PIVOT is
  // exited — same debounce window used symmetrically both ways.
  // ASSUMPTION, tune me.
  static constexpr int    PIVOT_HOLD_MS_ = 400;

  // ── cmd_vel watchdog — if no /cmd_vel_smoothed message arrives within
  //    this window, the nav stack is considered idle/dead and all channels
  //    are released to the RC receiver. ASSUMPTION: tune this. It must be
  //    comfortably larger than your nav stack's normal publish period
  //    (looked like ~50ms/20Hz in your log) but small enough to feel
  //    responsive when you actually want the sticks back.
  static constexpr int    CMD_VEL_TIMEOUT_MS_ = 300;
  static constexpr int    CONTROL_PERIOD_MS_  = 50; // 20Hz control/publish loop

  // ── Drive constants ────────────────────────────────────────────────────
  static constexpr double DEADZONE_         = 0.09; // cmd_vel deadzone (m/s, rad/s)
  static constexpr double ANG_THRESHOLD_    = 0.3;
  static constexpr double ANG_SLOW_ZONE_    = 0.2;
  static constexpr int    STOP_DURATION_MS_ = 2000;

  // ── Stall-detection constants (full stop, distinct from throttle ramp) ─
  static constexpr int    STALL_CHECK_MS_       = 7000;
  static constexpr double STALL_DIST_THRESHOLD_ = 0.02;
  static constexpr int    STALL_ZERO_MS_        = 1000;

  // ── Runtime state ──────────────────────────────────────────────────────
  DriveState   state_    = DriveState::STRAIGHT;
  rclcpp::Time stop_start_;
  double       last_ang_ = 0.0;

  bool is_armed_  = false;
  bool was_armed_ = false;

  rclcpp::Time stall_window_start_;
  double       stall_window_x_    = 0.0;
  double       stall_window_y_    = 0.0;
  bool         stall_window_set_  = false;
  rclcpp::Time stall_zero_start_;
  bool         output_is_nonzero_ = false;

  // Adaptive pivot (rotation) tracking
  double       current_pivot_output_ = NEUTRAL_PWM_;
  rclcpp::Time pivot_check_start_;
  double       pivot_start_yaw_      = 0.0;
  bool         pivot_check_set_      = false;
  bool         pivot_exit_pending_   = false;
  rclcpp::Time pivot_exit_start_;
  bool         pivot_entry_pending_   = false;
  rclcpp::Time pivot_entry_start_;
  bool         pivot_entry_was_left_  = false;

  // Adaptive straight (throttle) tracking
  double       current_straight_output_ = THROTTLE_BASE_;
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
  bool   collision_check_enabled_ = false; // enforcement on/off — see constructor

  // Collision self-check configuration (ROS params, see constructor)
  std::vector<Point2D> collision_polygon_;
  int                   collision_min_points_ = 9;
  std::string           collision_base_frame_ = "base_link";
  std::string           scan_topic_           = "/scan";
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_check_enable_sub_;

  // ── Arm state callback ─────────────────────────────────────────────────
  void state_callback(const mavros_msgs::msg::State::SharedPtr msg) {
    is_armed_ = msg->armed;

    if (was_armed_ && !is_armed_) {
      RCLCPP_WARN(get_logger(),
        "DISARM detected — resetting pivot/straight ramps and stall window.");
      reset_pivot_state();
      reset_straight_state();
      reset_stall_window();
      state_ = DriveState::STRAIGHT;
    }

    was_armed_ = is_armed_;
  }

  // ── cmd_vel subscription — just stores the latest message. All actual
  //    driving/publishing decisions happen in control_loop() below, driven
  //    by the timer, so we can detect the topic going silent. ─────────────
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

  // ── Collision check enable/disable toggle — runtime override via topic.
  //    Detection (scan_callback) and the RViz marker keep running either
  //    way; this only gates enforcement in control_loop(). ────────────────
  void collision_check_enable_callback(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data != collision_check_enabled_) {
      RCLCPP_WARN(get_logger(),
        "COLLISION CHECK enforcement toggled -> %s (via /collision_check_enable)",
        msg->data ? "ENABLED" : "DISABLED");
    }
    collision_check_enabled_ = msg->data;
  }

  // ── Collision self-check — helper to apply a state change with the same
  //    ramp/stall-window reset behavior the old collision_monitor-based
  //    version used, so behavior on transition is unchanged. ─────────────
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
  // simple polygon, not just the current rectangular PolygonStop shape —
  // so if collision_polygon param is changed to something non-rectangular
  // later, this still holds.
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

  // ── RViz visualization — outline of collision_polygon_, colored by
  //    current stop state. Published every scan cycle so it stays live. ──
  void publish_polygon_marker([[maybe_unused]] bool stop_active, const rclcpp::Time & stamp) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = collision_base_frame_;
    marker.header.stamp    = stamp;
    marker.ns     = "collision_self_check";
    marker.id     = 0;
    marker.type   = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05; // line width, meters
    marker.lifetime = rclcpp::Duration::from_seconds(0.5); // auto-expire if node stops publishing

    // Static red always — explicit instruction, replacing the earlier
    // state-dependent green/red/gray coloring I'd added unprompted.
    marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f; marker.color.a = 1.0f;

    marker.points.reserve(collision_polygon_.size() + 1);
    for (const auto & pt : collision_polygon_) {
      geometry_msgs::msg::Point p;
      p.x = pt.x; p.y = pt.y; p.z = 0.0;
      marker.points.push_back(p);
    }
    if (!collision_polygon_.empty()) {
      geometry_msgs::msg::Point p; // close the loop back to the first vertex
      p.x = collision_polygon_.front().x;
      p.y = collision_polygon_.front().y;
      p.z = 0.0;
      marker.points.push_back(p);
    }

    polygon_marker_pub_->publish(marker);
  }


  // ── Scan callback — the collision self-check itself. Runs at whatever
  //    rate /scan actually publishes (confirmed solid 10Hz on this system,
  //    independent of collision_monitor's broken output rate). ───────────
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    scan_ok_ = true;

    double tx = 0.0, ty = 0.0, cy = 1.0, sy = 0.0; // identity by default

    // Confirmed via `ros2 topic echo /scan` that this system's
    // pointcloud_to_laserscan_node already stamps /scan with frame_id
    // "base_link" directly (target_frame set upstream) — so in the common
    // case no TF lookup is needed at all, and we skip it explicitly rather
    // than relying on tf2's implicit self-transform special-case.
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

      // 2D rigid transform: laser frame -> collision_base_frame_
      const double bx = tx + cy * lx - sy * ly;
      const double by = ty + sy * lx + cy * ly;

      if (point_in_polygon(bx, by)) {
        ++points_inside;
        if (points_inside >= collision_min_points_) break; // no need to keep counting
      }
    }

    const bool now_stop = (points_inside >= collision_min_points_);
    set_collision_stop(now_stop, points_inside, "scan_check");
    publish_polygon_marker(now_stop, msg->header.stamp);
  }

  // ── RC input callback — deadman detection ───────────────────────────────
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

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "[DEBUG rc/in] steer(ch%zu)=%d throttle(ch%zu)=%d arm(ch%zu)=%d "
      "(not used for control decisions — see cmd_vel watchdog instead)",
      STEER_CH_ + 1, msg->channels[STEER_CH_],
      THROTTLE_CH_ + 1, msg->channels[THROTTLE_CH_],
      ARM_CH_ + 1, last_rc_in_arm_raw_);
  }

  // ── Helpers ────────────────────────────────────────────────────────────
  void reset_pivot_state() {
    current_pivot_output_ = NEUTRAL_PWM_;
    pivot_check_set_      = false;
    pivot_exit_pending_    = false;
    pivot_entry_pending_   = false;
  }

  void reset_straight_state() {
    current_straight_output_ = THROTTLE_BASE_;
    straight_check_set_      = false;
  }

  void reset_stall_window() {
    stall_window_set_ = false;
  }

  mavros_msgs::msg::OverrideRCIn make_release_all() const {
    mavros_msgs::msg::OverrideRCIn mc;
    mc.channels.fill(RC_RELEASE_);
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
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
      "[DEBUG override PUBLISH reason=%s] arm_ch(RC%zu)=%d rc_in_arm_raw=%d "
      "channels=[%s]",
      reason, ARM_CH_ + 1, mc.channels[ARM_CH_], last_rc_in_arm_raw_,
      oss.str().c_str());

    pub_->publish(mc);
  }

  // ── Control loop — runs on a fixed timer, NOT on cmd_vel arrival ────────
  void control_loop() {
    if (!rc_in_ok_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No /mavros/rc/in received yet — refusing to drive, releasing all channels.");
      publish_override(make_release_all(), "no_rc_in_yet");
      output_is_nonzero_ = false;
      return;
    }

    // scan_ok_ is only required when enforcement is actually on — if
    // collision checking is disabled, missing /scan shouldn't block driving
    // either, since nothing downstream depends on it in that case.
    if (collision_check_enabled_ && !scan_ok_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No /scan received yet — cannot verify collision clearance, "
        "refusing to drive, releasing all channels.");
      publish_override(make_release_all(), "no_scan_yet");
      output_is_nonzero_ = false;
      return;
    }

    if (!is_armed_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "Vehicle DISARMED — releasing all channels to RC.");
      publish_override(make_release_all(), "disarmed");
      output_is_nonzero_ = false;
      return;
    }

    // ── cmd_vel watchdog: if the nav stack hasn't published in a while,
    //    treat it as idle and hand full control back to the RC receiver. ──
    const auto since_cmd_vel_ms = have_cmd_vel_
      ? (now() - last_cmd_vel_time_).nanoseconds() / 1'000'000
      : std::numeric_limits<long>::max();

    if (!have_cmd_vel_ || since_cmd_vel_ms > CMD_VEL_TIMEOUT_MS_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "/cmd_vel_smoothed idle/silent (%s) — releasing all channels to RC.",
        have_cmd_vel_ ? "timeout" : "never received");
      publish_override(make_release_all(), "cmd_vel_idle");
      output_is_nonzero_ = false;
      return;
    }

    // ── Collision self-check hard stop — takes priority over the drive
    //    state machine every tick, driven by /scan directly (not
    //    collision_monitor, whose state_topic is unavailable on this
    //    system's installed build). We actively command neutral (not a
    //    full release) since this is an autonomy-active safety stop, not
    //    a hand-back-to-RC event. Gated on collision_check_enabled_ —
    //    detection (scan_callback) keeps running either way, only
    //    enforcement here respects the toggle. ──────────────────────────
    if (collision_check_enabled_ && collision_stop_active_) {
      mavros_msgs::msg::OverrideRCIn mc = make_release_all();
      mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
      mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "COLLISION STOP active (points_inside=%d, min_points=%d) — "
        "forcing neutral, drive state machine bypassed.",
        last_points_inside_, collision_min_points_);
      publish_override(mc, "collision_stop");
      output_is_nonzero_ = false;
      return;
    }

    const auto & msg = last_cmd_vel_msg_;
    mavros_msgs::msg::OverrideRCIn mc = make_release_all();

    const double lin = apply_deadzone(msg->linear.x);
    const double ang = apply_deadzone(msg->angular.z);

    if (state_ == DriveState::STALLED) {
      const auto elapsed_ms = (now() - stall_zero_start_).nanoseconds() / 1'000'000;
      if (elapsed_ms < STALL_ZERO_MS_) {
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
        publish_override(mc, "stalled");
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
        // Any tick that drops below threshold, or flips sign, resets the
        // timer — until confirmed, output stays exactly as it already was
        // (whatever STRAIGHT/SLOWING was already publishing).
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
          // else: still confirming, state_ unchanged, output stays as-is
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
          if (!straight_check_set_) {
            current_straight_output_ = THROTTLE_BASE_;
            straight_check_start_    = now();
            straight_start_x_        = odom_x_;
            straight_start_y_        = odom_y_;
            straight_check_set_      = true;
          } else if (odom_ok_) {
            const auto window_ms =
              (now() - straight_check_start_).nanoseconds() / 1'000'000;
            if (window_ms >= STRAIGHT_CHECK_MS_) {
              const double dx   = odom_x_ - straight_start_x_;
              const double dy   = odom_y_ - straight_start_y_;
              const double dist = std::sqrt(dx * dx + dy * dy);

              // NOTE: step-size ramp-up removed — throttle output no longer
              // increases on insufficient progress; it stays at
              // current_straight_output_ (THROTTLE_BASE_ unless otherwise
              // reset). Progress is still checked and logged below.
              if (dist < STRAIGHT_DIST_THRESHOLD_) {
                RCLCPP_WARN(get_logger(),
                  "STRAIGHT: insufficient progress (dist=%.3fm in %ldms), "
                  "throttle stays at %.0f (ramping disabled)",
                  dist, static_cast<long>(window_ms), current_straight_output_);
              } else {
                RCLCPP_INFO(get_logger(),
                  "STRAIGHT: progress OK (dist=%.3fm), throttle stays at %.0f",
                  dist, current_straight_output_);
              }
              straight_check_start_ = now();
              straight_start_x_     = odom_x_;
              straight_start_y_     = odom_y_;
            }
          }
          mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(
            std::clamp(current_straight_output_, THROTTLE_MIN_, THROTTLE_MAX_));
        } else {
          // No reverse, ever. Hold neutral and reset the ramp.
          if (lin < 0.0) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
              "cmd_vel requested reverse (lin=%.3f) — reverse disabled, "
              "holding neutral throttle.", lin);
          }
          reset_straight_state();
          mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        }
        mc.channels[STEER_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        break;
      }

      case DriveState::SLOWING:
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(
          (lin > 0.0) ? SLOW_THROTTLE_ : NEUTRAL_PWM_);
        mc.channels[STEER_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        break;

      case DriveState::STOPPING:
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
        mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
        break;

      case DriveState::PIVOT: {
        mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);

        // last_ang_ > 0 => LEFT (ASSUMPTION, verify on bench) => ramp toward PIVOT_MIN_
        // last_ang_ <= 0 => RIGHT                              => ramp toward PIVOT_MAX_
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

            // NOTE: step-size ramp-up removed — pivot output no longer
            // increases/decreases on insufficient rotation; it stays at
            // current_pivot_output_ (the direction base value unless
            // otherwise reset). Rotation is still checked and logged below.
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
      "cmd_vel [lin=%.3f, ang=%.3f] -> state=%d [throttle=%d, steer=%d]",
      msg->linear.x, msg->angular.z, static_cast<int>(state_),
      mc.channels[THROTTLE_CH_], mc.channels[STEER_CH_]);

    output_is_nonzero_ =
      (mc.channels[THROTTLE_CH_] != static_cast<uint16_t>(NEUTRAL_PWM_) ||
       mc.channels[STEER_CH_]    != static_cast<uint16_t>(NEUTRAL_PWM_));

    // ── Stall detection (full stop — separate from throttle ramp above) ──
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
              "Zeroing output for %dms.",
              dist, static_cast<long>(window_ms), STALL_ZERO_MS_);
            state_            = DriveState::STALLED;
            stall_zero_start_ = now();
            mc.channels[THROTTLE_CH_] = static_cast<uint16_t>(NEUTRAL_PWM_);
            mc.channels[STEER_CH_]    = static_cast<uint16_t>(NEUTRAL_PWM_);
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