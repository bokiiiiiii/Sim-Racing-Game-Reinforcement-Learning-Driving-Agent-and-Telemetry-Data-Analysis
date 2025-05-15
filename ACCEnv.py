import gymnasium as gym  # Use Gymnasium
import numpy as np
import struct
import time
import keyboard
from gymnasium import spaces  # Use Gymnasium spaces
from ACCTelemetry import ACCTelemetry  # Import ACCTelemetry
import ACCController as acc_ctrl  # Import ACCController


class ACCEnv(gym.Env):  # This will now refer to gymnasium.Env
    """
    Assetto Corsa Competizione (ACC) Reinforcement Learning Environment.

    This environment interfaces with ACC to allow an RL agent to learn driving policies.
    It handles:
    - Reading telemetry data from ACC's shared memory.
    - Sending control inputs (steering, throttle, brake) to the game.
    - Calculating rewards based on driving performance.
    - Resetting the game session for new episodes.
    """

    metadata = {"render_modes": [], "render_fps": 30}

    # --- Normalization Constants ---
    MAX_SPEED_KMH = 350.0
    MAX_GEARS = (
        8.0  # Max gear number + R + N (e.g., 7 gears -> (7+1)/8 for normalization)
    )
    MAX_RPM_DEFAULT = 10000.0  # Default max RPM if not found in static data
    MAX_TYRE_TEMP_C = 120.0
    CAR_WORLD_POS_NORMALIZATION = 1000.0  # Factor to normalize car world position

    # --- Reset Behavior ---
    INITIAL_THROTTLE_VALUE = 70  # Percentage for initial throttle burst
    INITIAL_THROTTLE_DURATION = 0.2  # Seconds for initial throttle burst
    RESET_MENU_DELAY = 5  # Seconds to wait for menu navigation during reset

    # --- Action Smoothing ---
    DEFAULT_ACTION_SMOOTHING_FACTOR = 0.3

    # --- Episode Settings ---
    DEFAULT_MAX_EPISODE_STEPS = 10000
    REWARD_PRINT_INTERVAL = 500  # Print reward every N steps

    # --- Reward Coefficients (can be tuned) ---
    REWARD_SPEED_FACTOR = 1.0 / 10.0  # Smaller denominator = larger reward per km/h
    REWARD_PROGRESS_MULTIPLIER = 200.0
    PENALTY_OFF_TRACK = -100.0
    REWARD_SURVIVAL = 0.01
    PENALTY_DAMAGE_MULTIPLIER = -100.0
    PENALTY_SLIP_MULTIPLIER = -10.0
    PENALTY_STUCK_OFF_TRACK_QUALIFYING = -500.0  # Specific penalty applied in step()

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
                    return float(val.item())  # .item() extracts Python scalar
                print(
                    f"Warning: Expected a single scalar value or single-element ndarray, but received ndarray: {val}. Using default value: {default_if_error}"
                )
                return default_if_error
            if isinstance(val, str):
                # Remove null characters from the string
                val = val.replace("\x00", "")
                # Check if the string matches the time format MM:SS:mmm
                if ":" in val:
                    try:
                        parts = val.split(":")
                        if len(parts) == 3:
                            minutes = int(parts[0])
                            seconds = int(parts[1])
                            milliseconds = int(parts[2])
                            # Convert to total seconds
                            return float(minutes * 60 + seconds + milliseconds / 1000.0)
                        elif len(parts) == 2:  # Handle SS:mmm format if it exists
                            seconds = int(parts[0])
                            milliseconds = int(parts[1])
                            return float(seconds + milliseconds / 1000.0)
                    except ValueError:
                        # If parsing as time fails, fall through to general float conversion
                        pass
                # Attempt general float conversion for other strings
                return float(val)
            return float(val)  # For Python scalars, numpy scalars, 0-dim arrays
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
        reward_config=None,  # Optional dictionary to override default reward coefficients
    ):
        super(ACCEnv, self).__init__()

        # Action space: [steer, throttle, brake]
        # steer: -1 (left) to 1 (right)
        # throttle: 0 (none) to 1 (full)
        # brake: 0 (none) to 1 (full)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: 15 features (see _get_observation for details)
        # Features include speed, steering angle, pedal inputs, gear, RPM, track position,
        # off-track status, damage levels, tyre slip, tyre temperatures, and car world position.
        self.observation_shape = (15,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32
        )

        # Initialize ACC Telemetry
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

        # --- RL State Variables (updated from telemetry) ---
        self.speed_kmh = 0.0
        self.steer_angle = 0.0
        self.throttle_input = 0.0  # Actual throttle from game physics
        self.brake_input = 0.0  # Actual brake from game physics
        self.gear = 0
        self.rpm = 0.0
        self.normalized_car_position = 0.0
        self.current_lap_time_ms = 0.0
        self.last_lap_time_ms = 0.0
        self.car_world_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.is_off_track = False
        self.suspension_damage = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.aero_damage = 0.0
        self.engine_damage = 0.0
        self.tyre_slip = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.tyre_core_temperature = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.session_type = 0  # E.g., Practice, Qualifying, Race
        self.game_status = 0  # E.g., AC_LIVE, AC_PAUSE

        # --- Helper Variables ---
        self.previous_normalized_car_position = 0.0
        self.previous_total_damage = 0.0
        self.current_episode_steps = 0
        self.max_episode_steps = max_episode_steps

        self.action_smoothing_factor = action_smoothing_factor
        self.previous_applied_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_applied_action = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )  # Agent's raw action (before smoothing)

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
        }
        if reward_config:
            self.reward_coeffs.update(reward_config)

        self.steps_since_last_reward_print = 0

    def step(self, action):
        """Execute an action and return the new state"""

        # Parse action (Steer, Throttle, Brake)
        steer, throttle, brake = action

        # Control the vehicle (via keyboard simulation or vJoy)
        self._apply_action(steer, throttle, brake)

        # Read new state
        obs = self._get_observation()

        # Calculate Reward
        reward = self._calculate_reward()

        # Periodically print the current step's reward
        self.steps_since_last_reward_print += 1
        if self.steps_since_last_reward_print >= self.REWARD_PRINT_INTERVAL:
            print(f"Reward at step {self.current_episode_steps + 1}: {reward:.4f}")
            self.steps_since_last_reward_print = 0

        # Check if terminated or truncated
        terminated, truncated = self._check_termination()
        self.done = (
            terminated or truncated
        )  # Update self.done for internal logic if needed

        # If terminated due to specific conditions, adjust reward.
        condition_for_reward_adjustment = (
            terminated and self.speed_kmh == 0.0 and self.status == 2
        )
        if condition_for_reward_adjustment:
            reward += self.reward_coeffs["stuck_off_track_qualifying_penalty"]
            print(
                f"Adjusted reward by {-self.reward_coeffs['stuck_off_track_qualifying_penalty']} due to termination condition (speed 0, status 2)."
            )

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):  # Add seed and options parameters
        """Reset the environment and return the initial observation"""
        super().reset(seed=seed)  # Call super for Gymnasium compatibility

        acc_ctrl.reset()
        keyboard.press_and_release("esc")
        keyboard.press_and_release("up")
        keyboard.press_and_release("up")
        keyboard.press_and_release("enter")
        keyboard.press_and_release("left")
        keyboard.press_and_release("enter")
        time.sleep(self.RESET_MENU_DELAY)
        keyboard.press_and_release("enter")
        time.sleep(self.RESET_MENU_DELAY)  # Wait for game to load into session

        # Apply a short throttle burst to get the car moving
        print("Applying initial throttle burst...")
        try:
            acc_ctrl.set_throttle(self.INITIAL_THROTTLE_VALUE)
            time.sleep(self.INITIAL_THROTTLE_DURATION)
            acc_ctrl.set_throttle(0)  # Release throttle
            print("Initial throttle burst complete.")
        except Exception as e:
            print(f"Error applying initial throttle burst: {e}")

        # Reset internal RL-related state variables
        # These will be overwritten by data from ACC in _get_observation()
        self.speed_kmh = 0.0
        self.steer_angle = 0.0
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.gear = 0
        self.rpm = 0
        self.normalized_car_position = 0.0
        self.current_lap_time_ms = 0
        # self.last_lap_time_ms is usually not cleared at reset to compare the first lap
        self.car_world_position = np.array([0.0, 0.0, 0.0])
        self.is_off_track = False
        self.suspension_damage = np.array([0.0, 0.0, 0.0, 0.0])
        self.aero_damage = 0.0
        self.engine_damage = 0.0
        self.tyre_slip = np.array([0.0, 0.0, 0.0, 0.0])
        self.tyre_core_temperature = np.array([0.0, 0.0, 0.0, 0.0])

        self.previous_normalized_car_position = 0.0  # Reset progress tracking
        self.previous_total_damage = 0.0  # Reset damage tracking
        self.current_episode_steps = 0
        self.previous_applied_action = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )  # Reset smoothed action

        self.done = False

        # Read initial state
        # At reset, usually, the game is reset first (if possible), then observations are read
        obs = self._get_observation()
        # Update previous_normalized_car_position for reward calculation of step 0
        self.previous_normalized_car_position = self.normalized_car_position
        self.previous_total_damage = (
            np.mean(self.suspension_damage) + self.aero_damage + self.engine_damage
        ) / 3.0

        print("Environment has been reset.")
        return obs, {}  # Gymnasium reset returns obs, info

    def _apply_action(self, steer, throttle, brake):
        """Apply action to the game (via ACCController controlling vJoy)"""
        current_raw_action = np.array([steer, throttle, brake], dtype=np.float32)

        # Apply action smoothing
        smoothed_action = (
            self.action_smoothing_factor * current_raw_action
            + (1 - self.action_smoothing_factor) * self.previous_applied_action
        )

        smooth_steer, smooth_throttle, smooth_brake = smoothed_action

        # Convert action values from RL environment range to the range expected by ACCController functions
        # steer: [-1, 1] -> [-100, 100]
        controller_steer = smooth_steer * 100.0
        # throttle: [0, 1] -> [0, 100]
        controller_throttle = smooth_throttle * 100.0
        # brake: [0, 1] -> [0, 100]
        controller_brake = smooth_brake * 100.0

        # Use functions in ACCController to control vJoy
        try:
            acc_ctrl.set_steering(controller_steer)
            acc_ctrl.set_throttle(controller_throttle)
            acc_ctrl.set_brake(controller_brake)
        except Exception as e:
            print(f"Error applying action (vJoy): {e}")
            # Error handling logic can be added here, e.g., try to reinitialize vJoy or flag environment error

        # Update internal record of control state
        self.last_applied_action = smoothed_action  # Store the smoothed action
        self.previous_applied_action = (
            smoothed_action  # Update for next step's smoothing
        )

        # print(f"Raw Action: Steer={steer:.2f}, Throttle={throttle:.2f}, Brake={brake:.2f}")
        # print(f"Smoothed Action: Steer={smooth_steer:.2f} (->{controller_steer:.0f}), Throttle={smooth_throttle:.2f} (->{controller_throttle:.0f}), Brake={smooth_brake:.2f} (->{controller_brake:.0f})")

    # Note: _scale_action was previously commented out and is confirmed to be unused. Removing it.

    def _get_observation(self):
        """
        Reads game state from ACC Shared Memory and processes it into a normalized observation vector.

        The observation vector (self.observation_shape = (15,)) consists of the following features in order:
        0. speed_kmh_normalized: Speed in km/h, normalized by MAX_SPEED_KMH.
        1. steer_angle: Steering angle, assumed to be in [-1, 1] from telemetry.
        2. throttle_input: Actual throttle pedal input from game physics [0, 1].
        3. brake_input: Actual brake pedal input from game physics [0, 1].
        4. gear_normalized: Current gear, normalized. (R=-1, N=0, 1st=1, ... then (gear+1)/MAX_GEARS).
        5. rpm_normalized: Engine RPM, normalized by self.max_rpm.
        6. normalized_car_position: Lap completion percentage [0, 1].
        7. is_off_track: Boolean (1.0 if off-track, 0.0 if on-track).
        8. suspension_damage_avg: Average suspension damage across all wheels [0, 1].
        9. aero_damage: Aerodynamic damage [0, 1].
        10. engine_damage: Engine damage [0, 1].
        11. tyre_slip_avg: Average tyre slip across all wheels. (Raw value, may need further scaling/clipping depending on typical range).
        12. tyre_core_temp_avg_normalized: Average tyre core temperature, normalized by MAX_TYRE_TEMP_C.
        13. car_world_pos_x_normalized: Car's X world coordinate, normalized by CAR_WORLD_POS_NORMALIZATION.
        14. car_world_pos_z_normalized: Car's Z world coordinate (often forward/backward on track plane), normalized by CAR_WORLD_POS_NORMALIZATION.

        Returns:
            np.ndarray: The normalized observation vector.
        """
        try:
            acc_data = self.acc_telemetry.getACCData()
        except Exception as e:
            print(f"Error reading ACC data: {e}")
            # Return a zero vector or the last valid observation in case of error
            # Returning a zero vector here to ensure dimension matching
            return np.zeros(self.observation_shape, dtype=np.float32)

        # Update internal state variables
        self.speed_kmh = self._ensure_scalar_float(acc_data.get("speedKmh", 0.0))
        self.steer_angle = self._ensure_scalar_float(acc_data.get("steerAngle", 0.0))
        self.throttle_input = self._ensure_scalar_float(acc_data.get("gas", 0.0))
        self.brake_input = self._ensure_scalar_float(acc_data.get("brake", 0.0))

        raw_gear = acc_data.get(
            "gear", 1
        )  # Default to Neutral (1) before subtracting 1
        scalar_gear = self._ensure_scalar_float(raw_gear, default_if_error=1)
        self.gear = (
            int(scalar_gear) - 1
        )  # ACC: 0=R, 1=N, 2=1st... Gym: -1=R, 0=N, 1=1st...

        self.rpm = self._ensure_scalar_float(acc_data.get("rpms", 0.0))
        self.normalized_car_position = self._ensure_scalar_float(
            acc_data.get("normalizedCarPosition", 0.0)
        )

        # current_lap_time_ms and last_lap_time_ms are not directly in obs_list, but ensure scalar if used elsewhere for calculations
        self.current_lap_time_ms = self._ensure_scalar_float(
            acc_data.get("currentTime", 0)
        )
        self.last_lap_time_ms = self._ensure_scalar_float(acc_data.get("lastTime", 0))

        # Ensure car_world_position is a flat numpy array of 3 floats
        raw_car_world_pos = acc_data.get("carWorldPosition", [0.0, 0.0, 0.0])
        try:
            self.car_world_position = np.array(
                raw_car_world_pos, dtype=np.float32
            ).flatten()
            if self.car_world_position.size != 3:
                # print(f"Warning: carWorldPosition after flatten does not have size 3: {self.car_world_position}. Using default.")
                self.car_world_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        except Exception as e:
            # print(f"Warning: Could not process carWorldPosition '{raw_car_world_pos}': {e}. Using default.")
            self.car_world_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.is_off_track = bool(
            self._ensure_scalar_float(
                acc_data.get("isOffTrack", 0.0), default_if_error=0.0
            )
        )

        # Ensure damage and tyre data are flat numpy arrays for np.mean
        raw_susp_damage = acc_data.get("suspensionDamage", [0.0] * 4)
        try:
            self.suspension_damage = np.array(
                raw_susp_damage, dtype=np.float32
            ).flatten()
            if self.suspension_damage.size != 4:
                self.suspension_damage = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.suspension_damage = np.array([0.0] * 4, dtype=np.float32)

        self.aero_damage = self._ensure_scalar_float(acc_data.get("aeroDamage", 0.0))
        self.engine_damage = self._ensure_scalar_float(
            acc_data.get("engineDamage", 0.0)
        )

        raw_tyre_slip = acc_data.get("tyreSlip", [0.0] * 4)
        try:
            self.tyre_slip = np.array(raw_tyre_slip, dtype=np.float32).flatten()
            if self.tyre_slip.size != 4:
                self.tyre_slip = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.tyre_slip = np.array([0.0] * 4, dtype=np.float32)

        raw_tyre_temp = acc_data.get("tyreCoreTemperature", [0.0] * 4)
        try:
            self.tyre_core_temperature = np.array(
                raw_tyre_temp, dtype=np.float32
            ).flatten()
            if self.tyre_core_temperature.size != 4:
                self.tyre_core_temperature = np.array([0.0] * 4, dtype=np.float32)
        except:
            self.tyre_core_temperature = np.array([0.0] * 4, dtype=np.float32)

        # Ensure self.session_type is updated, not self.status for session type checks
        self.session_type = int(
            self._ensure_scalar_float(acc_data.get("SessionType", 0))
        )  # Default to 0 if not found

        self.status = int(
            self._ensure_scalar_float(acc_data.get("status", 0.0))
        )  # Keep reading game status (Live, Pause, etc.)

        # Construct observation vector (obs)
        # All elements must be scalar floats.
        obs_list = [
            self._ensure_scalar_float(self.speed_kmh / self.MAX_SPEED_KMH),
            self._ensure_scalar_float(
                self.steer_angle
            ),  # Assuming steerAngle is already in [-1, 1] range from telemetry
            self._ensure_scalar_float(
                self.throttle_input
            ),  # Actual throttle from game physics
            self._ensure_scalar_float(
                self.brake_input
            ),  # Actual brake from game physics
            self._ensure_scalar_float(
                (self.gear + 1) / self.MAX_GEARS
            ),  # Normalize gear: R=-1, N=0, 1st=1...
            self._ensure_scalar_float(
                self.rpm / self.max_rpm if self.max_rpm > 0 else 0.0
            ),
            self._ensure_scalar_float(self.normalized_car_position),
            self._ensure_scalar_float(1.0 if self.is_off_track else 0.0),
            self._ensure_scalar_float(
                np.mean(self.suspension_damage)
                if self.suspension_damage.size > 0
                else 0.0
            ),
            self._ensure_scalar_float(self.aero_damage),
            self._ensure_scalar_float(self.engine_damage),
            self._ensure_scalar_float(
                np.mean(self.tyre_slip) if self.tyre_slip.size > 0 else 0.0
            ),  # May need clipping/scaling if range is large
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
            ),  # Using Z for "forward" on track plane
        ]

        try:
            obs = np.array(obs_list, dtype=np.float32)
        except ValueError as e:
            print(f"Critical Error: Could not convert obs_list to np.array. Error: {e}")
            # You might want to inspect individual elements of obs_list here if the error persists
            # The loop below is already included in the original code, keeping it for clarity
            for i, item in enumerate(obs_list):
                print(f"Debug: obs_list[{i}] (type: {type(item)}): {item}")
            return np.zeros(
                self.observation_shape, dtype=np.float32
            )  # Return a safe default with shape (15,)

        # The obs_list creates a (15,) array. Return it directly.
        # DummyVecEnv will add the batch dimension (1,).
        # Removed the reshape block as DummyVecEnv expects (15,)

        if obs.shape != self.observation_shape:  # Check against (15,)
            print(
                f"Warning: Actual observation vector dimension ({obs.shape}) does not match expected shape ({self.observation_shape}). Please check _get_observation or observation_space definition."
            )
            # Attempt to return a zero array of the correct shape as a fallback
            return np.zeros(
                self.observation_shape, dtype=np.float32
            )  # Return shape (15,)

        return obs

    def _calculate_reward(self):
        """Calculate reward value based on current state"""
        reward = 0.0
        # Get current total damage for comparison
        current_total_damage = (
            np.mean(self.suspension_damage) + self.aero_damage + self.engine_damage
        ) / 3.0

        # 1. Speed reward
        reward += self.speed_kmh * self.reward_coeffs["speed_factor"]

        # 2. Track progress reward
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
            self.speed_kmh > 10
        ):  # Only reward progress at a certain speed to avoid farming points by moving slowly
            reward += progress_made * self.reward_coeffs["progress_multiplier"]

        # 3. Off-track penalty
        if self.is_off_track:
            reward += self.reward_coeffs["off_track_penalty"]  # Penalty is negative

        # 4. Time penalty (or survival reward)
        reward += self.reward_coeffs["survival_reward"]

        # 5. Damage penalty
        damage_increase = current_total_damage - self.previous_total_damage
        if damage_increase > 0.005:  # Tolerate very small floating point errors
            reward += (
                damage_increase * self.reward_coeffs["damage_penalty_multiplier"]
            )  # Multiplier is negative

        # 6. Excessive slip penalty
        avg_slip = (
            np.mean(self.tyre_slip)
            if isinstance(self.tyre_slip, (list, np.ndarray))
            and len(self.tyre_slip) > 0
            else 0.0
        )
        # Penalize if average slip is above a threshold (e.g., 0.5)
        slip_threshold = 0.5
        if avg_slip > slip_threshold:
            reward += (avg_slip - slip_threshold) * self.reward_coeffs[
                "slip_penalty_multiplier"
            ]  # Multiplier is negative

        # 7. Reward for completing a lap (if last_lap_time_ms is updated)
        #    This requires more reliable lap completion detection, ACC's lastTime may not update instantly
        # if self.current_lap_time_ms < 1000 and self.last_lap_time_ms > 0: # Just crossed the line
        # new_lap_completed = True # Assume new lap completion detected
        # if new_lap_completed:
        #    reward += 500.0 # Large reward
        #    if self.last_lap_time_ms < self.best_lap_time_ms: # Assume self.best_lap_time_ms exists
        #        reward += (self.best_lap_time_ms - self.last_lap_time_ms) / 100.0 # Extra reward for breaking record

        # Update variables for next frame's calculation
        self.previous_normalized_car_position = self.normalized_car_position
        self.previous_total_damage = current_total_damage

        return float(reward)

    def _check_termination(self):
        """Check if termination or truncation conditions are met"""
        terminated = False
        truncated = False  # Initialize truncated flag
        self.current_episode_steps += 1

        # 1. Severe damage (Termination condition)
        current_total_damage = (
            np.mean(self.suspension_damage) + self.aero_damage + self.engine_damage
        ) / 3.0
        if current_total_damage > 0.8:  # Total average damage exceeds 80%
            print(f"Terminating: Vehicle severely damaged ({current_total_damage:.2f})")
            terminated = True

        # 2. Off-track for extended period (optional, if reward function isn't enough to bring it back)
        # if self.is_off_track:
        #     self.time_off_track_counter = getattr(self, 'time_off_track_counter', 0) + 1
        #     if self.time_off_track_counter > 150: # Approx 5 seconds (assuming 30FPS)
        #         print("Terminating: Off-track for too long")
        #         terminated = True
        # else:
        #     self.time_off_track_counter = 0

        # Terminate if speed is zero and game status is AC_LIVE (2).
        # User's original termination condition.
        if self.speed_kmh == 0.0 and self.status == 2:
            print("Terminating episode: Speed is 0 and game status is AC_LIVE (2).")
            terminated = True

        # 3. Reached maximum steps (Truncation condition)
        if (
            not terminated and self.current_episode_steps >= self.max_episode_steps
        ):  # Check only if not already terminated
            print(f"Truncating: Reached maximum steps ({self.max_episode_steps})")
            truncated = True

        # 4. Abnormal car state (e.g., game not running or connection lost - handled in _get_observation)
        # Could also consider terminating if acc_data fails to read multiple times

        # Old lap_distance logic can be removed as we now use normalizedCarPosition
        # if hasattr(self, "lap_distance") and self.lap_distance > 5000:
        #     print("Terminating: Reached simulated track length (old logic)")
        #     terminated = True # This would be a custom termination

        return terminated, truncated  # Return both flags

    def close(self):
        """Clean up environment resources"""
        print("Closing ACCEnv...")
        if hasattr(self, "acc_telemetry") and self.acc_telemetry:
            self.acc_telemetry.stop()
        # Consider if vJoy controller also needs to be reset
        try:
            acc_ctrl.reset()  # Reset vJoy inputs
            print("vJoy controller has been reset.")
        except Exception as e:
            print(f"Error resetting vJoy controller: {e}")
        print("ACCEnv closed.")
