import gymnasium as gym
import numpy as np
import torch
import librosa
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
from typing import Tuple

# ========================================
# 1. Beat Tracker (Real-Time Music Sync)
# ========================================
class BeatTracker:
    def __init__(self, audio_file):
        self.y, self.sr = librosa.load(audio_file, sr=None)
        _, beat_frames = librosa.beat.beat_track(y=self.y, sr=self.sr, units='frames')
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        self.current_beat_idx = 0

    def update(self, current_time: float) -> Tuple[float, int]:
        # Advance beat index if passed
        while (self.current_beat_idx < len(self.beat_times) and
               current_time >= self.beat_times[self.current_beat_idx]):
            self.current_beat_idx += 1

        # Compute beat interval and phase
        if len(self.beat_times) > 1:
            interval = np.mean(np.diff(self.beat_times))
        else:
            interval = 0.5  # fallback
        phase = (current_time % interval) / interval if interval > 0 else 0.0
        beat_idx = self.current_beat_idx % 4  # 4-count cycle
        return phase, int(beat_idx)

# ========================================
# 2. Robot Dance Environment
# ========================================
class RobotDanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, beat_tracker=None, max_episode_steps=2000):
        super().__init__()
        self.render_mode = render_mode
        self.beat_tracker = beat_tracker
        self.max_episode_steps = max_episode_steps

        # Create underlying humanoid env
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space  # Box(-1,1, shape=(17,))

        # Time setup
        self.step_time = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        self.elapsed_time = 0.0
        self.step_count = 0
        self.prev_action = np.zeros(self.action_space.shape)

        # Choreography: list of target poses (12 DoF for arms/legs)
        self.all_poses = [
            np.zeros(12),
            np.array([0.5, 0.5, -0.3] * 4),
            np.array([-0.5, 0.5, 0.0] * 4),
            np.array([0.0, -0.5, 0.3] * 4),
        ]
        self.current_poses = self.all_poses[:1]  # Start simple

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.elapsed_time = 0.0
        self.step_count = 0
        self.prev_action = np.zeros_like(self.prev_action)
        self.current_beat_idx = 0
        if self.beat_tracker:
            self.beat_tracker.current_beat_idx = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed_time += self.step_time
        self.step_count += 1

        # Get beat info
        phase, beat_idx = 0.0, 0
        if self.beat_tracker:
            phase, beat_idx = self.beat_tracker.update(self.elapsed_time)

        try:
            joint_angles = obs[7:19]  # Approximate upper/lower body joints
        except IndexError:
            joint_angles = np.zeros(12)

        # Pose matching
        target = self.current_poses[beat_idx % len(self.current_poses)]
        pose_reward = -np.linalg.norm(joint_angles - target)

        # Smoothness
        smooth_reward = -np.linalg.norm(action - self.prev_action)

        # Stability (low COM velocity)
        stability_reward = -np.linalg.norm(obs[3:6])  # linear velocity of torso

        # Beat phase alignment (optional: encourage movement on beat)
        beat_reward = -abs(phase - 0.0) if phase < 0.2 or phase > 0.8 else -0.5

        total_reward = (
            1.0 * pose_reward +
            0.3 * smooth_reward +
            0.2 * stability_reward +
            0.1 * beat_reward
        )

        self.prev_action = action.copy()

        # Termination
        terminated = terminated  # from humanoid (e.g., fall)
        truncated = self.step_count >= self.max_episode_steps

        return obs, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# ========================================
# 3. Curriculum Callback (Add poses over time)
# ========================================
class CurriculumCallback(BaseCallback):
    def __init__(self, env, eval_freq=50000, verbose=1):
        super().__init__(verbose)
        self.env = env.unwrapped  # unwrap DummyVecEnv
        self.eval_freq = eval_freq
        self.last_pose_add = 0

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        if (steps >= self.last_pose_add + 50000 and 
            len(self.env.current_poses) < len(self.env.all_poses)):
            self.env.current_poses.append(
                self.env.all_poses[len(self.env.current_poses)]
            )
            self.last_pose_add = steps
            print(f"[Curriculum] Added pose {len(self.env.current_poses)}/"
                  f"{len(self.env.all_poses)} at step {steps}")

        if steps % 50000 == 0:
            lr = self.model.learning_rate(self.progress_remaining)
            print(f"LR: {lr:.5f}, Progress: {100*(1-self.progress_remaining):.0f}%")

        return True

# ========================================
# 4. Learning Rate Schedule
# ========================================
def lr_schedule(progress):
    return 2e-5 + progress * (3e-4 - 2e-5)  # decreases as progress → 0

# ========================================
# 5. Main Training
# ========================================
if __name__ == "__main__":
    SEED = 42
    MUSIC_PATH = "path_to_your_music_file.mp3"  # Replace with your music file
    MODEL_PATH = "single_dancer_400k"
    VIDEO_FOLDER = "eval_videos"
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    print("Loading beat tracker...")
    beat_tracker = BeatTracker(MUSIC_PATH)

    print("Creating single robot environment...")
    env = RobotDanceEnv(beat_tracker=beat_tracker, max_episode_steps=2000)
    vec_env = DummyVecEnv([lambda: env])

    curriculum_cb = CurriculumCallback(vec_env)

    print("Starting training (400k steps)...")
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=lr_schedule,
        buffer_size=100000,
        batch_size=256,
        ent_coef="auto",
        tau=0.02,
        gamma=0.99,
        train_freq=64,
        gradient_steps=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        seed=SEED,
    )

    try:
        model.learn(
            total_timesteps=400_000,
            callback=curriculum_cb,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        model.save(MODEL_PATH)
        print(f"Model saved: {MODEL_PATH}.zip")

    env.close()

    # ================================
    # 🎥 Record Final Dance Video
    # ================================
    print("🎥 Recording final dance...")
    eval_env = RobotDanceEnv(
        render_mode="rgb_array",
        beat_tracker=BeatTracker(MUSIC_PATH),
        max_episode_steps=2000
    )
    eval_env = RecordVideo(
        eval_env,
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda x: True,
        name_prefix="final_dance"
    )

    obs, _ = eval_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = eval_env.step(action)

    eval_env.close()
    print(f"Final video saved in: {VIDEO_FOLDER}")



