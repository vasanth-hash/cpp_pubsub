// line_follower_rc_bridge.cpp
//
// Bridge: /cmd_vel (geometry_msgs/Twist, from the red line follower)
//      -> /mavros/rc/override (mavros_msgs/OverrideRCIn)
//
// Drives a SKID-STEER robot that can only (a) translate forward/back or
// (b) rotate in place -- never both at once. This is deliberately an
// "Option B" (stop-and-turn) bridge:
//
//   * If the steering command |angular.z| exceeds ang_threshold *and stays
//     there long enough* (see confirmation below), the robot STOPS
//     translating and rotates in place until the correction shrinks.
//   * Otherwise, if linear.x > 0, it drives straight forward.
//   * Otherwise it holds neutral.
//
// Rotation always wins over forward. Throttle and steer channels are never
// mixed. When switching between FORWARD and ROTATE, a short neutral "settle"
// is inserted so the chassis physically stops before changing motion type.
//
// ── ROTATE CONFIRMATION (debounce / hysteresis) ─────────────────────────
// A single noisy angular.z sample must NOT be allowed to trigger the settle
// and stop the robot. So entering (and leaving) ROTATE is debounced:
//
//   ENTER: |ang| must stay above ang_threshold, in the SAME direction,
//          continuously for rotate_confirm_ms before we commit to a turn.
//          During that window the robot keeps driving forward/holding --
//          it does NOT stop. Any tick below threshold, or a direction flip,
//          resets the timer. A noise spike therefore never reaches settle.
//
//   EXIT:  once rotating, |ang| must stay BELOW ang_threshold continuously
//          for rotate_exit_confirm_ms before we leave the turn. This rejects
//          the symmetric failure: a noisy low sample mid-turn briefly kicking
//          the robot out of ROTATE and firing a spurious settle. Set
//          rotate_exit_confirm_ms=0 for instant exit.
//
// While rotating, the turn DIRECTION is latched at commit time; only the
// turn SPEED tracks live |ang|. This stops a single sign-flipped noise tick
// from momentarily commanding the opposite steer direction.
//
// NOTE: the ROTATE above is the *Option-B, cmd_vel-driven* stop-and-turn for
// lane corrections. It is DISTINCT from the 180° spin-in-place feature added
// below (rotate-in-place flag), which is commanded out-of-band by a Bool and
// is NOT driven by cmd_vel. See the two blocks flagged "180 SPIN" below.
//
// ── 180 SPIN: rotate-in-place flag (/bridge/rotate_in_place) ─────────────
// The follower owns the 180° turn and its vision-based completion. During
// that spin it DELIBERATELY stops publishing /cmd_vel (publishing only a
// Bool). If we did nothing, the /cmd_vel watchdog below would trip and hand
// control back to the physical RC mid-spin. So this bridge subscribes to a
// std_msgs/Bool on /bridge/rotate_in_place: while that flag is true AND fresh
// (see spin_flag_timeout_ms_), the bridge drives RC1 to a fixed spin_pwm_
// (1700) and RC3 to neutral, BYPASSING the /cmd_vel watchdog. The staleness
// check means a crashed follower can't leave it spinning forever. The spin
// PWM lives HERE, not in the follower, so the bridge stays the SOLE publisher
// to /mavros/rc/override (no two-publisher race).
//
// ── ARM-WARMUP ────────────────────────────────────────────────────────────
// On every disarmed->armed edge, hold RC1/RC3 at neutral for arm_warmup_sec
// before resuming any cmd_vel / spin conversion, so the robot never lurches
// the instant it is armed.
//
// ── PRECEDENCE in the control loop (the important part) ───────────────────
//   safety gates (rc_in, armed) -> arm-warmup -> rotate flag ->
//   cmd_vel watchdog -> Option-B conversion.
// Rotate sits AHEAD of the cmd_vel watchdog (cmd_vel is silent during the
// spin) but BEHIND the safety gates (so it still won't spin when disarmed or
// during warmup).
//
// HARDWARE FACTS (from the operator, all overridable as params):
//   neutral (both channels resting) ............. 1501
//   forward .................................... hard jump to forward_pwm (no ramp)
//   rotate left  ............................... lower PWM = faster
//   rotate right ............................... higher PWM = faster
//   between the left/right bases the yaw channel is a DEAD ZONE (no rotation)
//   180 spin ................................... fixed spin_pwm (1700) on RC1
//
// The ARM channel (RC6) is NEVER overridden -- it is always released (0),
// so arming/disarming is 100% controlled by the physical remote.
//
// RC channel map (0-indexed): RC1=steer, RC3=throttle, RC6=arm.

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <mavros_msgs/msg/override_rc_in.hpp>
#include <mavros_msgs/msg/rc_in.hpp>
#include <mavros_msgs/msg/state.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

class LineFollowerRcBridge : public rclcpp::Node {
public:
  LineFollowerRcBridge() : Node("line_follower_rc_bridge") {
    // ── Topics ────────────────────────────────────────────────────────────
    cmd_vel_topic_  = declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    override_topic_ = declare_parameter<std::string>("override_topic", "/mavros/rc/override");
    spin_flag_topic_ = declare_parameter<std::string>("spin_flag_topic", "/bridge/rotate_in_place");

    // ── Channel map (0-indexed: RC1=0, RC3=2, RC6=5) ─────────────────────
    steer_ch_    = declare_parameter<int>("steer_channel",    0); // RC1 rotation
    throttle_ch_ = declare_parameter<int>("throttle_channel", 2); // RC3 forward
    arm_ch_      = declare_parameter<int>("arm_channel",      5); // RC6 arm (never touched)

    // ── PWM levels ────────────────────────────────────────────────────────
    neutral_pwm_ = declare_parameter<int>("neutral_pwm", 1501);
    forward_pwm_ = declare_parameter<int>("forward_pwm", 1535); // hard jump, no ramp

    // Rotation: value AT the dead-zone edge = slowest turn. As |ang| grows
    // from ang_threshold toward ang_max, output moves *span* further out.
    // Left decreases (base -> base-span), right increases (base -> base+span).
    left_base_   = declare_parameter<int>("left_base_pwm",  1420);
    right_base_  = declare_parameter<int>("right_base_pwm", 1580);
    left_span_   = declare_parameter<int>("left_span_pwm",  40); // how far past the edge at full command
    right_span_  = declare_parameter<int>("right_span_pwm", 40);
    left_min_    = declare_parameter<int>("left_min_pwm",  1280); // hard floor (fastest left allowed)
    right_max_   = declare_parameter<int>("right_max_pwm", 1720); // hard ceiling (fastest right allowed)

    // ── 180 SPIN: fixed spin PWM + freshness window ───────────────────────
    spin_pwm_            = declare_parameter<int>("spin_pwm", 1660);           // RC1 during the flag-driven 180 spin
    spin_flag_timeout_ms_ = declare_parameter<int>("spin_flag_timeout_ms", 300); // flag older than this => treat as false (crashed follower can't spin forever)

    // ── Command interpretation ────────────────────────────────────────────
    ang_threshold_ = declare_parameter<double>("ang_threshold", 0.35); // enter ROTATE above this |ang|
    ang_max_       = declare_parameter<double>("ang_max", 1.0);       // |ang| that reaches full span (= follower's max_angular_z)
    deadzone_lin_  = declare_parameter<double>("deadzone_lin", 0.02);
    deadzone_ang_  = declare_parameter<double>("deadzone_ang", 0.02);
    invert_steer_  = declare_parameter<bool>("invert_steer", false);  // flip if bench turns the wrong way

    // ── Rotate confirmation (debounce) ────────────────────────────────────
    rotate_confirm_ms_      = declare_parameter<int>("rotate_confirm_ms", 500);     // |ang| must persist this long to ENTER rotate
    rotate_exit_confirm_ms_ = declare_parameter<int>("rotate_exit_confirm_ms", 0); // must stay below threshold this long to LEAVE rotate (0 = instant)

    // ── Timing / safety ───────────────────────────────────────────────────
    control_rate_hz_    = declare_parameter<double>("control_rate_hz", 50.0);
    cmd_vel_timeout_ms_ = declare_parameter<int>("cmd_vel_timeout_ms", 300); // silent -> release to RC
    settle_ms_          = declare_parameter<int>("settle_ms", 600);          // neutral pause on FORWARD<->ROTATE switch
    require_armed_      = declare_parameter<bool>("require_armed", true);     // only drive when /mavros/state armed
    require_rc_in_      = declare_parameter<bool>("require_rc_in", true);     // only drive once /mavros/rc/in is alive
    arm_warmup_sec_     = declare_parameter<double>("arm_warmup_sec", 2.0);   // neutral hold after each disarmed->armed edge

    // ── Pub / sub ─────────────────────────────────────────────────────────
    pub_ = create_publisher<mavros_msgs::msg::OverrideRCIn>(override_topic_, 10);

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, 10,
      std::bind(&LineFollowerRcBridge::cmdVelCb, this, std::placeholders::_1));

    spin_flag_sub_ = create_subscription<std_msgs::msg::Bool>(
      spin_flag_topic_, 10,
      std::bind(&LineFollowerRcBridge::spinFlagCb, this, std::placeholders::_1));

    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      "/mavros/state", 10,
      std::bind(&LineFollowerRcBridge::stateCb, this, std::placeholders::_1));

    rc_in_sub_ = create_subscription<mavros_msgs::msg::RCIn>(
      "/mavros/rc/in", 10,
      std::bind(&LineFollowerRcBridge::rcInCb, this, std::placeholders::_1));

    auto period = std::chrono::duration<double>(1.0 / control_rate_hz_);
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&LineFollowerRcBridge::controlLoop, this));

    RCLCPP_WARN(get_logger(), "=== LineFollowerRcBridge BUILD: %s %s ===", __DATE__, __TIME__);
    RCLCPP_INFO(get_logger(),
      "Bridge started (Option B: stop-and-turn). cmd_vel=%s override=%s spin_flag=%s | "
      "steer=RC%d throttle=RC%d arm=RC%d(ALWAYS RELEASED) | "
      "neutral=%d forward=%d spin=%d(timeout %dms) | "
      "left_base=%d(-%d,floor %d) right_base=%d(+%d,ceil %d) | "
      "ang_threshold=%.2f ang_max=%.2f invert_steer=%s | "
      "rotate_confirm=%dms rotate_exit_confirm=%dms | "
      "settle=%dms timeout=%dms rate=%.0fHz warmup=%.1fs require_armed=%s require_rc_in=%s",
      cmd_vel_topic_.c_str(), override_topic_.c_str(), spin_flag_topic_.c_str(),
      steer_ch_ + 1, throttle_ch_ + 1, arm_ch_ + 1,
      neutral_pwm_, forward_pwm_, spin_pwm_, spin_flag_timeout_ms_,
      left_base_, left_span_, left_min_, right_base_, right_span_, right_max_,
      ang_threshold_, ang_max_, invert_steer_ ? "true" : "false",
      rotate_confirm_ms_, rotate_exit_confirm_ms_,
      settle_ms_, cmd_vel_timeout_ms_, control_rate_hz_, arm_warmup_sec_,
      require_armed_ ? "true" : "false", require_rc_in_ ? "true" : "false");
  }

private:
  enum class Mode { HOLD, FORWARD, ROTATE };

  // ── Callbacks (store only; all decisions happen in controlLoop) ─────────
  void cmdVelCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
    last_cmd_vel_      = msg;
    last_cmd_vel_time_ = now();
    have_cmd_vel_      = true;
  }

  // 180 SPIN flag from the follower. Store value + arrival time; freshness is
  // judged in the control loop so a crashed publisher can't pin us spinning.
  void spinFlagCb(const std_msgs::msg::Bool::SharedPtr msg) {
    spin_flag_value_ = msg->data;
    spin_flag_time_  = now();
    have_spin_flag_  = true;
  }

  void stateCb(const mavros_msgs::msg::State::SharedPtr msg) {
    is_armed_ = msg->armed;
  }

  void rcInCb(const mavros_msgs::msg::RCIn::SharedPtr msg) {
    rc_in_ok_ = true;
    (void)msg; // presence is all we need; arm channel stays with the physical RC
  }

  double deadzone(double v, double dz) const {
    return (std::fabs(v) < dz) ? 0.0 : v;
  }

  // Is the 180-spin flag currently commanding a spin? True only if we've
  // received it, its value is true, AND it's fresh within spin_flag_timeout_ms.
  bool spinActive() const {
    if (!have_spin_flag_ || !spin_flag_value_) return false;
    const long age_ms = (now() - spin_flag_time_).nanoseconds() / 1'000'000;
    return age_ms <= spin_flag_timeout_ms_;
  }

  // Map |ang| in [ang_threshold, ang_max] -> extra travel past the dead-zone
  // edge, as a 0..1 fraction. At the edge -> 0 (slowest); at ang_max -> 1.
  double outwardFraction(double abs_ang) const {
    double span = ang_max_ - ang_threshold_;
    if (span <= 1e-6) return 1.0;
    double f = (abs_ang - ang_threshold_) / span;
    return std::clamp(f, 0.0, 1.0);
  }

  // Steer PWM for a latched logical direction (left_logical = ang>0 before
  // invert) and a live magnitude. invert_steer is applied here.
  uint16_t yawPwmFor(bool left_logical, double abs_ang) const {
    bool left = invert_steer_ ? !left_logical : left_logical;
    double f = outwardFraction(abs_ang);
    double pwm;
    if (left) {
      pwm = left_base_ - f * left_span_;
      pwm = std::max(pwm, static_cast<double>(left_min_));
    } else {
      pwm = right_base_ + f * right_span_;
      pwm = std::min(pwm, static_cast<double>(right_max_));
    }
    return static_cast<uint16_t>(std::lround(pwm));
  }

  // All channels released (0 = "use the real RC input") except the ones we
  // explicitly set later. Arm channel is forced released in publishOverride.
  mavros_msgs::msg::OverrideRCIn releaseAll() const {
    mavros_msgs::msg::OverrideRCIn mc;
    mc.channels.fill(0);
    return mc;
  }

  void publishOverride(mavros_msgs::msg::OverrideRCIn mc) {
    mc.channels[arm_ch_] = 0; // arm channel is ALWAYS the physical remote's, never us
    pub_->publish(mc);
  }

  void publishNeutral() {
    mavros_msgs::msg::OverrideRCIn mc = releaseAll();
    mc.channels[throttle_ch_] = static_cast<uint16_t>(neutral_pwm_);
    mc.channels[steer_ch_]    = static_cast<uint16_t>(neutral_pwm_);
    publishOverride(mc);
  }

  // Drive RC1 to the fixed spin PWM, RC3 neutral. Pure in-place 180 spin.
  void publishSpin() {
    mavros_msgs::msg::OverrideRCIn mc = releaseAll();
    mc.channels[throttle_ch_] = static_cast<uint16_t>(neutral_pwm_);
    mc.channels[steer_ch_]    = static_cast<uint16_t>(spin_pwm_);
    publishOverride(mc);
  }

  // Begin a neutral settle, after which active_mode_ becomes `target`.
  void beginSettle(Mode target) {
    settling_      = true;
    settle_start_  = now();
    settle_target_ = target;
    RCLCPP_INFO(get_logger(), "Motion mode change -> %dms neutral settle.", settle_ms_);
    publishNeutral();
  }

  // ── Fixed-rate control loop ─────────────────────────────────────────────
  void controlLoop() {
    // ══ SAFETY GATES ════════════════════════════════════════════════════
    // Gate 1: receiver alive
    if (require_rc_in_ && !rc_in_ok_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "No /mavros/rc/in yet -- releasing all channels.");
      publishOverride(releaseAll());
      return;
    }

    // Gate 2: armed (arming itself is done from the physical remote).
    // Detect the disarmed->armed edge here to (re)start the arm-warmup hold.
    if (require_armed_ && !is_armed_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "DISARMED -- releasing all channels to RC.");
      publishOverride(releaseAll());
      resetToHold();
      was_armed_ = false;   // next arm will retrigger warmup
      return;
    }
    if (require_armed_ && is_armed_ && !was_armed_) {
      // disarmed -> armed edge
      was_armed_       = true;
      warmup_active_   = true;
      warmup_start_    = now();
      resetToHold();
      RCLCPP_INFO(get_logger(),
        "ARMED -- holding neutral for %.1fs arm-warmup before driving.", arm_warmup_sec_);
    }

    // ══ ARM-WARMUP ══════════════════════════════════════════════════════
    // Ahead of everything drive-related (rotate flag included) so nothing
    // moves in the first arm_warmup_sec after arming.
    if (warmup_active_) {
      const double elapsed = (now() - warmup_start_).seconds();
      if (elapsed < arm_warmup_sec_) {
        publishNeutral();
        return;
      }
      warmup_active_ = false;
      RCLCPP_INFO(get_logger(), "Arm-warmup complete -- resuming normal operation.");
    }

    // ══ 180 SPIN: rotate flag ════════════════════════════════════════════
    // Sits AHEAD of the cmd_vel watchdog because cmd_vel is intentionally
    // silent during the spin, but BEHIND the safety gates + warmup above.
    if (spinActive()) {
      // Leaving any Option-B state cleanly; the spin owns the chassis now.
      resetToHold();
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
        "180-SPIN active -- RC1=%d (bypassing cmd_vel watchdog).", spin_pwm_);
      publishSpin();
      return;
    }

    // ══ cmd_vel WATCHDOG ════════════════════════════════════════════════
    // If the follower went silent (and we're not spinning), hand control back
    // to the physical remote entirely.
    const long since_ms = have_cmd_vel_
      ? (now() - last_cmd_vel_time_).nanoseconds() / 1'000'000
      : std::numeric_limits<long>::max();
    if (!have_cmd_vel_ || since_ms > cmd_vel_timeout_ms_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "/cmd_vel idle (%s) -- releasing all channels to RC.",
        have_cmd_vel_ ? "timeout" : "never received");
      publishOverride(releaseAll());
      resetToHold();
      return;
    }

    // ══ OPTION-B CONVERSION ═════════════════════════════════════════════
    // ── Latest command ─────────────────────────────────────────────────────
    const double lin = deadzone(last_cmd_vel_->linear.x,  deadzone_lin_);
    const double ang = deadzone(last_cmd_vel_->angular.z, deadzone_ang_);
    const bool wants_rotate = (std::fabs(ang) > ang_threshold_);
    const bool this_left    = (ang > 0.0); // logical direction (pre-invert)

    // ── If mid-settle, hold neutral until it completes, then adopt target ──
    if (settling_) {
      const long ms = (now() - settle_start_).nanoseconds() / 1'000'000;
      if (ms < settle_ms_) {
        publishNeutral();
        return;
      }
      settling_    = false;
      active_mode_ = settle_target_;
      if (active_mode_ == Mode::ROTATE) {
        RCLCPP_INFO(get_logger(), "Settle done -> ROTATE (%s).",
                    rotate_left_ ? "LEFT" : "RIGHT");
      }
      // fall through to output
    } else {
      // ── Mode transition logic (only runs when not settling) ──────────────
      if (active_mode_ == Mode::ROTATE) {
        // Leaving ROTATE is debounced: |ang| must stay below threshold for
        // rotate_exit_confirm_ms continuously before we commit to exiting.
        if (wants_rotate) {
          exit_pending_ = false; // still turning, cancel any pending exit
        } else {
          if (rotate_exit_confirm_ms_ <= 0) {
            beginSettle(lin > 0.0 ? Mode::FORWARD : Mode::HOLD);
            exit_pending_ = false;
            return;
          }
          if (!exit_pending_) {
            exit_pending_       = true;
            exit_pending_start_ = now();
          } else {
            const long ex_ms = (now() - exit_pending_start_).nanoseconds() / 1'000'000;
            if (ex_ms >= rotate_exit_confirm_ms_) {
              exit_pending_ = false;
              beginSettle(lin > 0.0 ? Mode::FORWARD : Mode::HOLD);
              return;
            }
          }
        }
      } else {
        // In FORWARD or HOLD. Entering ROTATE is debounced: |ang| must stay
        // above threshold, same direction, for rotate_confirm_ms. During this
        // window we KEEP driving forward/holding -- we do NOT stop. This is
        // what makes a noise spike harmless.
        if (wants_rotate) {
          if (!rotate_pending_ || this_left != rotate_pending_left_) {
            rotate_pending_       = true;
            rotate_pending_start_ = now();
            rotate_pending_left_  = this_left;
          } else {
            const long en_ms = (now() - rotate_pending_start_).nanoseconds() / 1'000'000;
            if (en_ms >= rotate_confirm_ms_) {
              rotate_pending_ = false;
              rotate_left_    = rotate_pending_left_; // latch direction for the turn
              beginSettle(Mode::ROTATE);
              return;
            }
          }
          // else: still confirming -> active_mode_ unchanged, keep current output
        } else {
          rotate_pending_ = false; // dropped below threshold, cancel confirmation
          active_mode_    = (lin > 0.0) ? Mode::FORWARD : Mode::HOLD; // free switch, no settle
        }
      }
    }

    // ── Emit output for the active mode ────────────────────────────────────
    mavros_msgs::msg::OverrideRCIn mc = releaseAll();
    switch (active_mode_) {
      case Mode::ROTATE:
        mc.channels[throttle_ch_] = static_cast<uint16_t>(neutral_pwm_);      // pure spin in place
        mc.channels[steer_ch_]    = yawPwmFor(rotate_left_, std::fabs(ang));  // latched dir, live speed
        break;
      case Mode::FORWARD:
        mc.channels[throttle_ch_] = static_cast<uint16_t>(forward_pwm_);      // hard jump, no ramp
        mc.channels[steer_ch_]    = static_cast<uint16_t>(neutral_pwm_);
        break;
      case Mode::HOLD:
      default:
        mc.channels[throttle_ch_] = static_cast<uint16_t>(neutral_pwm_);
        mc.channels[steer_ch_]    = static_cast<uint16_t>(neutral_pwm_);
        break;
    }

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
      "cmd_vel[lin=%.3f ang=%.3f] mode=%s%s throttle=%d steer=%d",
      last_cmd_vel_->linear.x, last_cmd_vel_->angular.z,
      active_mode_ == Mode::ROTATE ? "ROTATE" :
      active_mode_ == Mode::FORWARD ? "FORWARD" : "HOLD",
      rotate_pending_ ? "(confirming)" : (exit_pending_ ? "(exit-confirming)" : ""),
      mc.channels[throttle_ch_], mc.channels[steer_ch_]);

    publishOverride(mc);
  }

  void resetToHold() {
    active_mode_    = Mode::HOLD;
    settling_       = false;
    rotate_pending_ = false;
    exit_pending_   = false;
  }

  // ── Params ──────────────────────────────────────────────────────────────
  std::string cmd_vel_topic_, override_topic_, spin_flag_topic_;
  int steer_ch_, throttle_ch_, arm_ch_;
  int neutral_pwm_, forward_pwm_;
  int left_base_, right_base_, left_span_, right_span_, left_min_, right_max_;
  int spin_pwm_, spin_flag_timeout_ms_;
  double ang_threshold_, ang_max_, deadzone_lin_, deadzone_ang_;
  bool invert_steer_;
  int rotate_confirm_ms_, rotate_exit_confirm_ms_;
  double control_rate_hz_;
  int cmd_vel_timeout_ms_, settle_ms_;
  bool require_armed_, require_rc_in_;
  double arm_warmup_sec_;

  // ── State ───────────────────────────────────────────────────────────────
  Mode active_mode_   = Mode::HOLD;
  Mode settle_target_ = Mode::HOLD;
  bool settling_ = false;
  rclcpp::Time settle_start_;

  // Entry confirmation
  bool rotate_pending_ = false;
  bool rotate_pending_left_ = false;
  rclcpp::Time rotate_pending_start_;
  bool rotate_left_ = false; // latched turn direction while in ROTATE

  // Exit confirmation
  bool exit_pending_ = false;
  rclcpp::Time exit_pending_start_;

  bool is_armed_ = false;
  bool rc_in_ok_ = false;

  // Arm-warmup
  bool was_armed_     = false;
  bool warmup_active_ = false;
  rclcpp::Time warmup_start_;

  // 180 SPIN flag
  bool have_spin_flag_  = false;
  bool spin_flag_value_ = false;
  rclcpp::Time spin_flag_time_;

  bool have_cmd_vel_ = false;
  rclcpp::Time last_cmd_vel_time_;
  geometry_msgs::msg::Twist::SharedPtr last_cmd_vel_;

  rclcpp::Publisher<mavros_msgs::msg::OverrideRCIn>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   cmd_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         spin_flag_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr     state_sub_;
  rclcpp::Subscription<mavros_msgs::msg::RCIn>::SharedPtr      rc_in_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LineFollowerRcBridge>());
  rclcpp::shutdown();
  return 0;
}