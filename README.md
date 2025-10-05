# BeatBot-Music-Synchronized-Robot-Dancing-with-Reinforcement-Learning

This project trains a **humanoid robot** to perform synchronized dance moves to music using **Reinforcement Learning (RL)**. The system leverages **Stable-Baselines3 (SAC)**, **MuJoCo’s Humanoid environment**, and a **Beat Tracker** powered by `librosa` for real-time music synchronization.

✨ Features
* **Beat Tracking**: Extracts beat timings from any audio file for rhythm-aware dancing.
* **Custom Dance Environment**: Extends `Humanoid-v5` with pose-based choreography and beat-phase rewards.
* **Curriculum Learning**: Gradually introduces new dance poses during training for smoother learning.
* **Reward Shaping**: Combines pose accuracy, smoothness, stability, and beat alignment.
* **Video Recording**: Captures the final dance sequence for visualization.

📂 Project Structure
├── dance.py                # Main training & evaluation script
├── README.md               # Documentation
├── requirements.txt        # Dependencies
├── eval_videos/            # Saved dance videos
└── single_dancer_400k.zip  # Trained model (after training)

🚀 Usage

1. Train the Model


python dance.py

* Trains for **400,000 steps** using **SAC**.
* Saves the model as `single_dancer_400k.zip`.

2. Evaluate & Record Dance

After training, the script will:

* Load the trained model.
* Run the robot with deterministic actions.
* Save a video in `eval_videos/final_dance-step-...mp4`.

## 🧠 Reward Function Breakdown

The total reward is a weighted sum of:

* **Pose Matching**: Distance from target choreography poses.
* **Smoothness**: Penalizes large action changes between steps.
* **Stability**: Encourages low center-of-mass velocity.
* **Beat Alignment**: Encourages movement near musical beats.

🎶 Customizing Music

* Place your audio file in the project directory.
* Update the `MUSIC_PATH` in `dance.py`:

📈 Curriculum Learning

* Starts with 1 base pose.
* Adds new poses every **50,000 steps**.
* Gradually increases movement complexity.

🎥 Example Output

After training, you’ll find videos in:
```
eval_videos/final_dance-episode-*.mp4
```
Showcasing the humanoid robot dancing in sync with the chosen song.

🔮 Future Improvements

* Multi-agent dance choreography.
* Style transfer for mimicking human dance datasets.
* Real-time beat adaptation during live playback.
* Integration with motion capture data for realistic poses.

📜 License

This project is released under the MIT License.
