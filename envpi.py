import torch
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedContinuous, DiscreteTensorSpec
from torchrl.envs import EnvBase


class NativeTorchRLCustomEnv(EnvBase):

  def __init__(self, device="cpu"):
    super().__init__(device=device)

    # Initialize your environment variables from envi.py
    self.start_x = 10.0
    self.min_x = 5.0
    self.max_x = 15.0
    self.max_step = 200

    # Define TorchRL specs
    self.enable_specs()

  def enable_specs(self):
    # Action space: Discrete (0 or 1)
    self.action_spec = DiscreteTensorSpec(
        n=2, shape=torch.Size([]), device=self.device, dtype=torch.long
    )

    # Observation space: 1-dimensional float vector
    self.observation_spec = Composite(
        observation=UnboundedContinuous(
            shape=torch.Size([1]), device=self.device, dtype=torch.float32
        )
    )

    # State spec (optional, but good practice for PPO)
    self.state_spec = self.observation_spec.clone()

  def _reset(self, tensordict=None):
    self.current_step = 0
    self.state = self.start_x

    obs = torch.tensor([self.state], dtype=torch.float32, device=self.device)
    return TensorDict(
        {"observation": obs}, batch_size=tensordict.batch_size if tensordict is not None else []
    )

  def _step(self, tensordict):
    # Extract action tensor and convert to scalar
    action = tensordict["action"].item()

    self.current_step += 1
    if action == 1:
      self.state += 0.5
    elif action == 0:
      self.state -= 0.5

    # Add random uniform noise matching envi.py logic
    self.state += float(torch.rand(1).item() * 1.6 - 0.8)

    terminated = bool(self.state < self.min_x or self.state > self.max_x)
    truncated = bool(self.current_step > self.max_step)
    done = terminated or truncated

    reward_val = 0.0 if terminated else 1.0
    reward = torch.tensor([reward_val], dtype=torch.float32, device=self.device)

    obs = torch.tensor([self.state], dtype=torch.float32, device=self.device)

    # Return the next tensordict structure required by TorchRL
    out = TensorDict(
        {
            "observation": obs,
            "reward": reward,
            "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
            "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
        },
        batch_size=tensordict.batch_size,
    )
    return out

  def _set_seed(self, seed: int):
    torch.manual_seed(seed)