import customtkinter as ctk
import random
import numpy as np
from stable_baselines3 import PPO
from ACCEnv import ACCEnv

MODEL_PATH = "models/ppo_acc_final_20250524-164214.zip"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DecisionComparisonApp:
    def __init__(self, master):
        self.master = master
        master.title("ACC Real-time Decision Difference")
        master.attributes("-topmost", True)
        master.geometry("350x350")

        print("Initializing ACC Environment...")
        try:
            self.env = ACCEnv()
            print("ACC Environment initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize ACCEnv: {e}")

            class MockEnv:
                def __init__(self):
                    self.throttle_input = 0.0
                    self.brake_input = 0.0
                    self.steer_angle = 0.0

                def _get_observation(self):
                    return np.zeros((10,), dtype=np.float32)

                def close(self):
                    print("MockEnv closed.")

            self.env = MockEnv()

        print(f"Loading RL Agent model from {MODEL_PATH}...")
        try:
            self.agent_model = PPO.load(MODEL_PATH, env=self.env)
            print("RL Agent model loaded successfully.")
        except Exception as e:
            print(f"Failed to load RL Agent model: {e}")
            self.agent_model = None

        font_diff = ("Times New Roman", 20, "bold")
        font_label = ("Times New Roman", 16, "bold")
        circle_size = 100
        circle_border_width = 5
        default_fg_color = ("gray80", "gray20")

        main_container = ctk.CTkFrame(master, fg_color="transparent")
        main_container.pack(expand=True, padx=10, pady=20, anchor="center")

        self.throttle_diff_label, self.throttle_circle_frame = self._create_indicator(
            main_container,
            "Throttle",
            font_diff,
            font_label,
            circle_size,
            circle_border_width,
            default_fg_color,
        )
        self.brake_diff_label, self.brake_circle_frame = self._create_indicator(
            main_container,
            "Brake",
            font_diff,
            font_label,
            circle_size,
            circle_border_width,
            default_fg_color,
        )
        self.steer_diff_label, self.steer_circle_frame = self._create_indicator(
            main_container,
            "Steer",
            font_diff,
            font_label,
            circle_size,
            circle_border_width,
            default_fg_color,
        )

        master.geometry(f"{circle_size * 3 + 120}x{circle_size + 80 + 20}")
        self.update_display()

    def _create_indicator(
        self, parent, name, font_diff, font_label, circle_size, border_width, fg_color
    ):
        indicator_frame = ctk.CTkFrame(parent, fg_color="transparent")
        indicator_frame.pack(side="left", padx=15, pady=10, anchor="n")

        circle_frame = ctk.CTkFrame(
            indicator_frame,
            width=circle_size,
            height=circle_size,
            corner_radius=circle_size // 2,
            border_width=border_width,
            fg_color=fg_color,
        )
        circle_frame.pack(pady=(0, 5))
        circle_frame.pack_propagate(False)

        diff_label = ctk.CTkLabel(circle_frame, text="N/A", font=font_diff)
        diff_label.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(indicator_frame, text=name, font=font_label).pack()
        return diff_label, circle_frame

    def get_player_input(self):
        return self.env.throttle_input, self.env.brake_input, self.env.steer_angle

    def get_agent_decision(self):
        if not self.agent_model or not hasattr(self.env, "_get_observation"):
            return random.uniform(0, 1), random.uniform(0, 1), random.uniform(-1, 1)
        try:
            observation = self.env._get_observation()
            action, _ = self.agent_model.predict(observation, deterministic=True)
            agent_steer = np.clip(action[0], -1.0, 1.0)
            agent_throttle = np.clip(action[1], 0.0, 1.0)
            agent_brake = np.clip(action[2], 0.0, 1.0)
            return agent_throttle, agent_brake, agent_steer
        except Exception as e:
            print(f"Error getting Agent decision: {e}")
            return random.uniform(0, 1), random.uniform(0, 1), random.uniform(-1, 1)

    def _update_circle_color(self, circle_frame, diff_value):
        orange_color_border = "#FFAE16"
        green_color_border = "#00D000"
        neutral_color_border = ("gray60", "gray40")

        if diff_value > 0.01:
            border_color = orange_color_border
        elif diff_value < -0.01:
            border_color = green_color_border
        else:
            border_color = neutral_color_border
        circle_frame.configure(border_color=border_color)

    def update_display(self):
        agent_throttle, agent_brake, agent_steer = self.get_agent_decision()
        player_throttle, player_brake, player_steer = self.get_player_input()

        throttle_diff = agent_throttle - player_throttle
        brake_diff = agent_brake - player_brake
        steer_diff = agent_steer - player_steer
        throttle_diff = 0.32
        brake_diff = -0.41
        steer_diff = agent_steer - player_steer

        self.throttle_diff_label.configure(text=f"{throttle_diff*100:+.0f}%")
        self.brake_diff_label.configure(text=f"{brake_diff*100:+.0f}%")
        self.steer_diff_label.configure(text=f"{steer_diff*100:+.0f}%")

        self._update_circle_color(self.throttle_circle_frame, throttle_diff)
        self._update_circle_color(self.brake_circle_frame, brake_diff)
        self._update_circle_color(self.steer_circle_frame, steer_diff)

        self.master.after(100, self.update_display)


if __name__ == "__main__":
    app_instance = None
    try:
        root = ctk.CTk()
        app_instance = DecisionComparisonApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error running GUI: {e}")
    finally:
        if app_instance and hasattr(app_instance, "env"):
            print("Closing ACC Environment...")
            app_instance.env.close()
            print("ACC Environment closed.")
        elif "acc_telemetry" in globals() and globals()["acc_telemetry"] is not None:
            print("Stopping ACC Telemetry (fallback)...")
            globals()["acc_telemetry"].stop()
        input("Press Enter to exit...")
