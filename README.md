# ACC Telemetry Data Analysis and Reinforcement Learning Driving Agent

This project aims to analyze telemetry data from Assetto Corsa Competizione (ACC) and train a Reinforcement Learning (RL) agent capable of autonomous driving within the game.

## Project Goals

*   **Telemetry Data Collection and Analysis**: 
Extract detailed telemetry data from ACC, such as speed, RPM, tire temperatures, suspension travel, etc.
*   **Environment Construction**: 
Build a simulation environment suitable for RL training based on ACC telemetry data.
*   **Reinforcement Learning Agent Training**: 
Train an RL agent using algorithms like Proximal Policy Optimization (PPO) to learn driving policies.

## Project Structure

*   `ACCController.py`: Script to control the ACC game.
*   `ACCEnv.py`: Defines the Reinforcement Learning environment.
*   `ACCTelemetry.py`: Script for processing ACC telemetry data.
*   `LocalParameters.py`: Local parameter settings.
*   `main.py`: Main execution file for the project.
*   `RealTimePlot.py`: Script for real-time plotting of telemetry data.
*   `requirements.txt`: Required Python packages for the project.
*   `RLtrain.py`: Reinforcement Learning training script.
*   `logs/`: Directory for storing training logs.
*   `models/`: Directory for storing trained models.

## Training Process Showcase

Below is a demonstration of the Reinforcement Learning agent's performance during training:

<!-- Insert your training process GIF here -->
![Training Process GIF](RL_Training.gif)

## How to Use

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start Training**:
    ```bash
    python RLtrain.py
    ```
