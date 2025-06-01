import gymnasium as gym
import numpy as np
import struct
import time
import keyboard
from gymnasium import spaces
from ACCTelemetry import ACCTelemetry
import ACCController as acc_ctrl


class ACCEnv(gym.Env):
    """
    Assetto Corsa Competizione (ACC) Reinforcement Learning Environment.

    This environment interfaces with ACC to allow an RL agent to learn driving policies, including:
    - Reading telemetry data from ACC's shared memory.
    - Sending control inputs (steering, throttle, brake) to ACC.
    - Calculating rewards based on driving performance.
    - Resetting the game session for new episodes.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    # --- Normalization Constants ---
    MAX_SPEED_KMH = 350.0
    MAX_GEARS = 8.0
    MAX_RPM_DEFAULT = 10000.0
    MAX_TYRE_TEMP_C = 120.0
    CAR_WORLD_POS_NORMALIZATION = 1000.0

    # --- Reset Behavior ---
    INITIAL_THROTTLE_VALUE = 70
    INITIAL_THROTTLE_DURATION = 2.5
    RESET_MENU_DELAY = 5

    # --- Action Smoothing ---
    DEFAULT_ACTION_SMOOTHING_FACTOR = 0.3

    # --- Episode Settings ---
    DEFAULT_MAX_EPISODE_STEPS = 50000
    REWARD_PRINT_INTERVAL = 500
    CUMULATIVE_REWARD_PRINT_INTERVAL = 100

    # --- Reward Coefficients ---
    REWARD_SPEED_FACTOR = 1.0 / 5.0
    REWARD_PROGRESS_MULTIPLIER = 200.0
    PENALTY_OFF_TRACK = -100.0
    REWARD_SURVIVAL = 0.01
    PENALTY_DAMAGE_MULTIPLIER = -100.0
    PENALTY_SLIP_MULTIPLIER = -30.0
    PENALTY_STUCK_OFF_TRACK_QUALIFYING = -10000.0
    PENALTY_STEERING_RATE = -1.0
    LOW_SPEED_PENALTY_FACTOR = 0.5

    # --- Reward Logic Thresholds ---
    DESIRED_MIN_SPEED_KMH = 15.0
    PROGRESS_REWARD_MIN_SPEED_KMH = 10.0
    DAMAGE_INCREASE_THRESHOLD = 0.005
    SLIP_THRESHOLD = 0.5
    STEERING_RATE_THRESHOLD = 0.1

    def _ensure_scalar_float(self, val, default_if_error=0.0):
        try:
            if isinstance(val, (list, tuple)):
                if len(val) == 1:
                    return float(val[0])
                print(
                    f"Warning: Expected a single scalar value or single-element list/tuple, but received list/tuple: {val}. Using default value: {default_if_error}"
                )
                return default_if_error
            elif isinstance(val, np.ndarray):
                if val.size == 1:
                    return float(val.item())
                print(
                    f"Warning: Expected a single scalar value or single-element ndarray, but received ndarray: {val}. Using default value: {default_if_error}"
                )
                return default_if_error
            if isinstance(val, str):
                val = val.replace("\x00", "")
                if ":" in val:
                    try:
                        parts = val.split(":")
                        # Handle MM:SS:mmm format
                        if len(parts) == 3:
                            minutes = int(parts[0])
                            seconds = int(parts[1])
                            milliseconds = int(parts[2])
                            return float(minutes * 60 + seconds + milliseconds / 1000.0)
                        # Handle SS:mmm format
                        elif len(parts) == 2:
                            seconds = int(parts[0])
                            milliseconds = int(parts[1])
                            return float(seconds + milliseconds / 1000.0)
                    except ValueError:
                        pass
                return float(val)
            return float(val)
        except (TypeError, ValueError) as e:
            print(
                f"Warning: Could not convert value '{val}' to scalar float due to: {e}. Using default value: {default_if_error}"
            )
            return default_if_error

    def __init__(
        self,
        acc_telemetry_instance=None,
        action_smoothing_factor=DEFAULT_ACTION_SMOOTHING_FACTOR,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        reward_config=None,
    ):
        super(ACCEnv, self).__init__()

        # Action space: [steer, throttle, brake]
        # steer:    -1 ~ 1  (left ~ right)
        # throttle:  0 ~ 1  (none ~ full)
        # brake:     0 ~ 1  (none ~ full)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space:
        # speed, steering angle, gear, RPM, track position,
        # suspension_damage_avg, tyre_slip_avg, tyre_core_temp_avg_normalized, car_world_pos_x_normalized, car_world_pos_z_normalized.
        self.observation_shape = (
            10,  # Reduced from 12 after removing throttle_input, brake_input
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32
        )

        self.acc_telemetry = (
            acc_telemetry_instance if acc_telemetry_instance else ACCTelemetry()
        )
        self.acc_telemetry.start()

        try:
            static_data = self.acc_telemetry.getstaticData()
            self.max_rpm = static_data.get("maxRpm", self.MAX_RPM_DEFAULT)
        except Exception as e:
            print(
                f"Warning: Failed to get static data during initialization: {e}. Using default max RPM."
            )
            self.max_rpm = self.MAX_RPM_DEFAULT

        # --- RL State Variables ---
        self.speed_kmh = 0.0
        self.steer_angle = 0.0
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.gear = 0
        self.rpm = 0.0
        self.normalized_car_position = 0.0
        self.current_lap_time_ms = 0.0
        self.last_lap_time_ms = 0.0
        self.car_world_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # self.is_off_track = False # Removed
        self.suspension_damage = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # self.aero_damage = 0.0 # Removed
        # self.engine_damage = 0.0 # Removed
        self.tyre_slip = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.tyre_core_temperature = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.session_type = 0
        self.game_status = 0

        # --- Helper Variables ---
        self.previous_normalized_car_position = 0.0
        self.previous_total_damage = 0.0
        self.current_episode_steps = 0
        self.max_episode_steps = max_episode_steps

        self.action_smoothing_factor = action_smoothing_factor
        self.previous_applied_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_applied_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.stuck_step_counter = 0
        self.MAX_STUCK_STEPS = (
            300  # Threshold for being stuck (e.g., 10 seconds at ~30 steps/sec)
        )

        self.done = False

        # Initialize reward coefficients, allowing overrides
        self.reward_coeffs = {
            "speed_factor": self.REWARD_SPEED_FACTOR,
            "progress_multiplier": self.REWARD_PROGRESS_MULTIPLIER,
            "off_track_penalty": self.PENALTY_OFF_TRACK,
            "survival_reward": self.REWARD_SURVIVAL,
            "damage_penalty_multiplier": self.PENALTY_DAMAGE_MULTIPLIER,
            "slip_penalty_multiplier": self.PENALTY_SLIP_MULTIPLIER,
            "stuck_off_track_qualifying_penalty": self.PENALTY_STUCK_OFF_TRACK_QUALIFYING,
            "steering_rate_penalty": self.PENALTY_STEERING_RATE,
            "low_speed_penalty_factor": self.LOW_SPEED_PENALTY_FACTOR,  # Add new factor to dict
        }
        if reward_config:
            self.reward_coeffs.update(reward_config)

    def step(self, action):
        """Execute an action and return the new state"""

        steer, throttle, brake = action  # Parse action

        self._apply_action(steer, throttle, brake)  # Control vehicle

        obs = self._get_observation()  # Read new state

        reward = self._calculate_reward()  # Calculate reward
        self.current_episode_reward += reward  # Accumulate reward for the episode

        terminated, truncated = (
            self._check_termination()
        )  # Check termination conditions
        self.done = terminated or truncated  # Update self.done for internal logic

        # If terminated due to specific conditions, adjust reward.
        condition_for_reward_adjustment = (
            terminated and self.speed_kmh == 0.0 and self.status == 2
        )
        if condition_for_reward_adjustment:
            reward += self.reward_coeffs["stuck_off_track_qualifying_penalty"]
            print(
                f"Adjusted reward by {self.reward_coeffs['stuck_off_track_qualifying_penalty']} due to termination condition (speed 0, status 2)."
            )

        if self.current_episode_steps % self.CUMULATIVE_REWARD_PRINT_INTERVAL == 0:
            print(
                f"Step {self.current_episode_steps:<6}: Cumulative Reward = {self.current_episode_reward:>9.0f}"
            )

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation"""
        super().reset(seed=seed)

        acc_ctrl.reset()
        keyboard.press_and_release("esc")
        keyboard.press_and_release("up")
        keyboard.press_and_release("up")
        keyboard.press_and_release("enter")
        keyboard.press_and_release("left")
        keyboard.press_and_release("enter")
        time.sleep(self.RESET_MENU_DELAY)
        keyboard.press_and_release("enter")
        time.sleep(self.RESET_MENU_DELAY)

        # Apply a short throttle burst to get the car moving
        print("Applying initial throttle burst...")
        try:
            acc_ctrl.set_throttle(self.INITIAL_THROTTLE_VALUE)
            time.sleep(self.INITIAL_THROTTLE_DURATION)
            acc_ctrl.set_throttle(0)
            print("Initial throttle burst complete.")
        except Exception as e:
            print(f"Error applying initial throttle burst: {e}")

        # Reset internal RL state variables
        self.speed_kmh = 0.0
        self.steer_angle = 0.0
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.gear = 0
        self.rpm = 0
        self.normalized_car_position = 0.0
        self.current_lap_time_ms = 0
        self.car_world_position = np.array([0.0, 0.0, 0.0])
        # self.is_off_track = False # Removed
        self.suspension_damage = np.array([0.0, 0.0, 0.0, 0.0])
        # self.aero_damage = 0.0 # Removed
        # self.engine_damage = 0.0 # Removed
        self.tyre_slip = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.tyre_core_temperature = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.previous_normalized_car_position = 0.0
        self.current_episode_steps = 0
        self.previous_applied_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.done = False
        self.current_episode_reward = (
            0.0  # Initialize cumulative reward for the episode
        )

        obs = self._get_observation()  # Read initial state
        self.previous_normalized_car_position = self.normalized_car_position
        self.previous_total_damage = (
            np.mean(self.suspension_damage) if self.suspension_damage.size > 0 else 0.0
        )

        print("Environment has been reset.")
        return obs, {}

    def _apply_action(self, steer, throttle, brake):
        """Apply action to the game via ACCController controlling vJoy"""
        current_raw_action = np.array([steer, throttle, brake], dtype=np.float32)

        # Apply action smoothing
        smoothed_action = (
            self.action_smoothing_factor * current_raw_action
            + (1 - self.action_smoothing_factor) * self.previous_applied_action
        )

        smooth_steer, smooth_throttle, smooth_brake = smoothed_action

        controller_steer = smooth_steer
        controller_throttle = smooth_throttle
        controller_brake = smooth_brake

        try:
            acc_ctrl.set_steering(controller_steer)
            acc_ctrl.set_throttle(controller_throttle)
            acc_ctrl.set_brake(controller_brake)
        except Exception as e:
            print(f"Error applying action (vJoy): {e}")

        self.last_applied_action = smoothed_action
        self.previous_applied_action = smoothed_action

    def _get_observation(self):
        """
        Reads game state from ACC Shared Memory and processes it into a normalized observation vector.

        The observation vector (self.observation_shape = (10,)) consists of the following features in order:
        0. speed_kmh_normalized: Speed in km/h, normalized by MAX_SPEED_KMH.
        1. steer_angle: Steering angle, assumed to be in [-1, 1] from telemetry.
        2. gear_normalized: Current gear, normalized. (R=-1, N=0, 1st=1, ... then (gear+1)/MAX_GEARS).
        3. rpm_normalized: Engine RPM, normalized by self.max_rpm.
        4. normalized_car_position: Lap completion percentage [0, 1].
        5. suspension_damage_avg: Average suspension damage across all wheels [0, 1].
        6. tyre_slip_avg: Average tyre slip across all wheels. (Raw value, may need further scaling/clipping depending on typical range).
        7. tyre_core_temp_avg_normalized: Average tyre core temperature, normalized by MAX_TYRE_TEMP_C.
        8. car_world_pos_x_normalized: Car's X world coordinate, normalized by CAR_WORLD_POS_NORMALIZATION.
        9. car_world_pos_z_normalized: Car's Z world coordinate (often forward/backward on track plane), normalized by CAR_WORLD_POS_NORMALIZATION.

        Returns:
            np.ndarray: The normalized observation vector.
        """
        try:
            acc_data = self.acc_telemetry.getACCData()
        except Exception as e:
            print(f"Error reading ACC data: {e}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        self.speed_kmh = self._ensure_scalar_float(acc_data.get("speedKmh", 0.0))
        self.steer_angle = self._ensure_scalar_float(acc_data.get("steerAngle", 0.0))
        self.throttle_input = self._ensure_scalar_float(
            acc_data.get("throttle", 0.0)
        )  # Correct based on LocalParameters
        self.brake_input = self._ensure_scalar_float(
            acc_data.get("brake", 0.0)
        )  # Correct

        raw_gear = acc_data.get("gear", 1)  # ACC: 0=R, 1=N, 2=1st... # Correct
        scalar_gear = self._ensure_scalar_float(raw_gear, default_if_error=1)
        self.gear = int(scalar_gear) - 1  # Gym: -1=R, 0=N, 1=1st...

        self.rpm = self._ensure_scalar_float(
            acc_data.get("rpm", 0.0)
        )  # Correct based on LocalParameters
        self.normalized_car_position = self._ensure_scalar_float(
            acc_data.get("normalizedCarPosition", 0.0)  # Correct
        )

        self.current_lap_time_ms = self._ensure_scalar_float(
            acc_data.get(
                "currentTime", 0
            )  # Correct (ACCTelemetry handles string conversion)
        )
        self.last_lap_time_ms = self._ensure_scalar_float(
            acc_data.get("lastTime", 0)
        )  # Correct (ACCTelemetry handles string conversion)

        # Get player car ID to correctly index into carCoordinates
        player_car_id = int(self._ensure_scalar_float(acc_data.get("playerCarID", 0)))
        all_car_coordinates = acc_data.get(
            "carCoordinates", []
        )  # This is a flat list: [x0,y0,z0, x1,y1,z1, ...]

        if isinstance(all_car_coordinates, list) and len(all_car_coordinates) > (
            player_car_id * 3 + 2
        ):
            car_x = all_car_coordinates[player_car_id * 3]
            car_y = all_car_coordinates[
                player_car_id * 3 + 1
            ]  # Y is typically up/down in simulators
            car_z = all_car_coordinates[
                player_car_id * 3 + 2
            ]  # Z is often forward/backward
            self.car_world_position = np.array([car_x, car_y, car_z], dtype=np.float32)
        else:
            self.car_world_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if not isinstance(all_car_coordinates, list):
                print(
                    f"Warning: carCoordinates is not a list: {type(all_car_coordinates)}"
                )
            elif len(all_car_coordinates) <= (player_car_id * 3 + 2):
                print(
                    f"Warning: carCoordinates list too short ({len(all_car_coordinates)}) for playerCarID {player_car_id}"
                )

        # self.is_off_track = bool( ... ) # Removed, as isOffTrack key is not reliably available

        # suspensionDamage is correctly converted by ACCTelemetry to a list
        raw_susp_damage = acc_data.get("suspensionDamage", [0.0] * 4)
        try:
            self.suspension_damage = np.array(
                raw_susp_damage, dtype=np.float32
            ).flatten()
            if self.suspension_damage.size != 4:  # Ensure it's 4 elements
                self.suspension_damage = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.suspension_damage = np.array([0.0] * 4, dtype=np.float32)

        # aeroDamage and engineDamage are not directly in LocalParameters with these names.
        # ACCTelemetry doesn't create them from carDamagefront etc. These will likely be 0.
        # self.aero_damage = self._ensure_scalar_float(acc_data.get("aeroDamage", 0.0)) # Removed
        # self.engine_damage = self._ensure_scalar_float(acc_data.get("engineDamage", 0.0)) # Removed

        # ACCTelemetry converts wheelSlipFL/FR/RL/RR to 'wheelSlip' list
        raw_tyre_slip = acc_data.get("wheelSlip", [0.0] * 4)
        try:
            self.tyre_slip = np.array(raw_tyre_slip, dtype=np.float32).flatten()
            if self.tyre_slip.size != 4:  # Ensure it's 4 elements
                self.tyre_slip = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.tyre_slip = np.array([0.0] * 4, dtype=np.float32)

        # ACCTelemetry converts TyreCoreTempFL/FR/RL/RR to 'TyreCoreTemp' list
        raw_tyre_temp = acc_data.get("TyreCoreTemp", [0.0] * 4)
        try:
            self.tyre_core_temperature = np.array(
                raw_tyre_temp, dtype=np.float32
            ).flatten()
            if self.tyre_core_temperature.size != 4:  # Ensure it's 4 elements
                self.tyre_core_temperature = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.tyre_core_temperature = np.array([0.0] * 4, dtype=np.float32)

        self.session_type = int(
            self._ensure_scalar_float(
                acc_data.get("session", 0)  # Changed from SessionType to session
            )  # 0: Unknown, 1: Practice, 2: Qualifying, ...
        )

        self.status = int(
            self._ensure_scalar_float(
                acc_data.get("status", 0.0)
            )  # 0: AC_OFF, 1: AC_REPLAY, 2: AC_LIVE, ...
        )

        # Construct the observation vector. All elements must be scalar floats.
        obs_list = [
            self._ensure_scalar_float(self.speed_kmh / self.MAX_SPEED_KMH),
            self._ensure_scalar_float(
                self.steer_angle
            ),  # Assumed to be in [-1, 1] from telemetry
            self._ensure_scalar_float(
                (self.gear + 1) / self.MAX_GEARS
            ),  # Normalized gear
            self._ensure_scalar_float(
                self.rpm / self.max_rpm if self.max_rpm > 0 else 0.0
            ),
            self._ensure_scalar_float(self.normalized_car_position),
            # Removed: self._ensure_scalar_float(1.0 if self.is_off_track else 0.0),
            self._ensure_scalar_float(
                np.mean(self.suspension_damage)
                if self.suspension_damage.size > 0
                else 0.0
            ),
            # Removed: self._ensure_scalar_float(self.aero_damage),
            # Removed: self._ensure_scalar_float(self.engine_damage),
            self._ensure_scalar_float(
                np.mean(self.tyre_slip) if self.tyre_slip.size > 0 else 0.0
            ),
            self._ensure_scalar_float(
                np.mean(self.tyre_core_temperature) / self.MAX_TYRE_TEMP_C
                if self.tyre_core_temperature.size > 0
                else 0.0
            ),
            self._ensure_scalar_float(
                self.car_world_position[0] / self.CAR_WORLD_POS_NORMALIZATION
                if self.car_world_position.size == 3
                else 0.0
            ),
            self._ensure_scalar_float(
                self.car_world_position[2] / self.CAR_WORLD_POS_NORMALIZATION
                if self.car_world_position.size == 3
                else 0.0
            ),
        ]

        try:
            obs = np.array(obs_list, dtype=np.float32)
        except ValueError as e:
            print(f"Critical Error: Could not convert obs_list to np.array. Error: {e}")
            for i, item in enumerate(obs_list):
                print(f"Debug: obs_list[{i}] (type: {type(item)}): {item}")
            return np.zeros(self.observation_shape, dtype=np.float32)

        if obs.shape != self.observation_shape:
            print(
                f"Warning: Actual observation vector dimension ({obs.shape}) does not match expected shape ({self.observation_shape})."
            )
            return np.zeros(
                self.observation_shape, dtype=np.float32
            )  # Fallback to correct shape

        return obs

    def _calculate_reward(self):
        """Calculate reward value based on current state"""
        reward = 0.0
        current_total_damage = (
            np.mean(self.suspension_damage) if self.suspension_damage.size > 0 else 0.0
        )

        # Speed reward / Low speed penalty
        if self.speed_kmh < self.DESIRED_MIN_SPEED_KMH:
            # Penalty increases as speed drops further below DESIRED_MIN_SPEED_KMH
            reward -= (
                self.DESIRED_MIN_SPEED_KMH - self.speed_kmh
            ) * self.reward_coeffs["low_speed_penalty_factor"]
        else:
            # Reward for speed when above DESIRED_MIN_SPEED_KMH
            reward += self.speed_kmh * self.reward_coeffs["speed_factor"]

        # Track progress reward
        progress_made = (
            self.normalized_car_position - self.previous_normalized_car_position
        )
        if progress_made < -0.8:  # Handle crossing finish line (0.99 -> 0.01)
            progress_made += 1.0
        elif (
            progress_made > 0.8
        ):  # Handle possible reverse crossing of finish line (0.01 -> 0.99), penalize
            progress_made -= 1.0

        if (
            self.speed_kmh > self.PROGRESS_REWARD_MIN_SPEED_KMH
        ):  # Only reward progress at a certain speed to avoid farming points by moving slowly
            reward += progress_made * self.reward_coeffs["progress_multiplier"]

        # Off-track penalty - Removed as self.is_off_track is removed
        # if self.is_off_track:
        #     reward += self.reward_coeffs["off_track_penalty"]  # Penalty is negative

        # Time penalty (or survival reward)
        reward += self.reward_coeffs["survival_reward"]

        # Damage penalty
        damage_increase = current_total_damage - self.previous_total_damage
        if (
            damage_increase > self.DAMAGE_INCREASE_THRESHOLD
        ):  # Tolerate very small floating point errors
            reward += (
                damage_increase * self.reward_coeffs["damage_penalty_multiplier"]
            )  # Multiplier is negative

        # Excessive slip penalty
        avg_slip = (
            np.mean(self.tyre_slip)
            if isinstance(self.tyre_slip, (list, np.ndarray))
            and len(self.tyre_slip) > 0
            else 0.0
        )
        # Penalize if average slip is above a threshold
        if avg_slip > self.SLIP_THRESHOLD:
            reward += (avg_slip - self.SLIP_THRESHOLD) * self.reward_coeffs[
                "slip_penalty_multiplier"
            ]  # Multiplier is negative

        steering_change = abs(
            self.last_applied_action[0] - self.previous_applied_action[0]
        )
        if steering_change > self.STEERING_RATE_THRESHOLD:
            reward += (
                steering_change - self.STEERING_RATE_THRESHOLD
            ) * self.reward_coeffs["steering_rate_penalty"]

        self.previous_normalized_car_position = self.normalized_car_position
        self.previous_total_damage = current_total_damage

        return float(reward)

    def _check_termination(self):
        """Check if termination or truncation conditions are met"""
        terminated = False
        truncated = False
        self.current_episode_steps += 1

        # Severe damage
        current_total_damage = (
            np.mean(self.suspension_damage) if self.suspension_damage.size > 0 else 0.0
        )
        if current_total_damage > 0.8:  # Total average damage exceeds 80%
            print(f"Terminating: Vehicle severely damaged ({current_total_damage:.2f})")
            terminated = True

        # Stuck or specific game state (e.g., speed is 0 when game is live)
        if (
            self.speed_kmh == 0.0 and self.status == 2
        ):  # status 2 is AC_LIVE (game is active)
            print("Terminating: Car off-track (speed is 0 and game is live).")
            terminated = True

        # Stuck for a prolonged period if speed is zero
        if not terminated:  # Only check if not already terminated by other conditions
            if self.speed_kmh == 0.0:
                self.stuck_step_counter += 1
            else:
                self.stuck_step_counter = 0  # Reset if car moves

            if self.stuck_step_counter > self.MAX_STUCK_STEPS:
                print(
                    f"Terminating: Car stuck (speed is 0 for {self.stuck_step_counter} steps)."
                )
                terminated = True

        # Reached maximum episode steps (Truncation)
        if not terminated and self.current_episode_steps >= self.max_episode_steps:
            print(f"Truncating: Reached maximum steps ({self.max_episode_steps}).")
            truncated = True

        return terminated, truncated

    def close(self):
        """Clean up environment resources."""
        print("Closing ACCEnv...")
        if hasattr(self, "acc_telemetry") and self.acc_telemetry:
            self.acc_telemetry.stop()
        try:
            acc_ctrl.reset()  # Reset vJoy inputs to neutral
            print("vJoy controller has been reset.")
        except Exception as e:
            print(f"Error resetting vJoy controller: {e}")
        print("ACCEnv closed.")
