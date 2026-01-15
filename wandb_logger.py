import wandb
import numpy as np

class WandBLogger:
    def __init__(self, project_name, run_name, config=None):
        """
        Initialize a WandB run.
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        wandb.init(project=project_name, name=run_name, config=config)
        
        self.episode_frames = []
        self.current_episode = 0

    def log_step(self, reward, loss=None, value=None, extra_info=None):
        """
        Log data at every step.
        """
        log_dict = {"reward": reward}
        if loss is not None:
            log_dict["loss"] = loss
        if value is not None:
            log_dict["value"] = value
        if extra_info:
            log_dict.update(extra_info)
        wandb.log(log_dict)

    def store_frame(self, frame):
        """
        Store a frame for video logging.
        Frame shape should be (H, W, C)
        """
        self.episode_frames.append(np.transpose(frame, (2, 0, 1)))  # C,H,W for wandb.Video

    def log_episode(self, episode_reward):
        """
        Log data at the end of an episode, including video if available.
        """
        log_dict = {
            "episode": self.current_episode,
            "episode_reward": episode_reward
        }
        if self.episode_frames:
            log_dict["gameplay_video"] = wandb.Video(
                np.array(self.episode_frames), fps=30, format="mp4"
            )
        wandb.log(log_dict)
        self.episode_frames = []  # Clear frames for next episode
        self.current_episode += 1

    def finish(self):
        wandb.finish()
