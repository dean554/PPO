import random
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

# action index -> (row delta, col delta): 0=up, 1=right, 2=down, 3=left
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class SnakeEnv(EnvBase):
    """Minimal snake on an N x N grid. Single (non-batched) environment.

    Observation (14 floats, all in [-1, 1]):
        head position (row, col)        normalized to [0, 1]   -> 2
        food position (row, col)        normalized to [0, 1]   -> 2
        food minus head (row, col)      normalized to [-1, 1]  -> 2
        danger flag for up/right/down/left (wall or body)      -> 4
        current direction, one-hot                             -> 4
    Reward: +1 eat food, -1 die (wall / self), -0.01 per step.
    Episode ends (terminated) on death, or on a full board (win).
    Episode is truncated if the snake goes `max_steps_without_food` steps
    without eating, which stops it circling forever.
    """

    def __init__(self, grid_size=20, max_steps_without_food=100,
                 device="cpu", seed=None):
        super().__init__(device=device, batch_size=torch.Size([]))
        self.n = grid_size
        self.max_idle = max_steps_without_food
        self._make_specs()
        self._set_seed(seed)

    # ---------- specs ----------
    def _make_specs(self):
        d = self.device
        self.observation_spec = Composite(
            observation=Bounded(low=-1.0, high=1.0, shape=(14,),
                                dtype=torch.float32, device=d),
            shape=self.batch_size,
        )
        # 4 discrete actions, one scalar integer per step
        self.action_spec = Categorical(4, shape=(), dtype=torch.long, device=d)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32, device=d)
        self.done_spec = Composite(
            done=Categorical(2, shape=(1,), dtype=torch.bool, device=d),
            terminated=Categorical(2, shape=(1,), dtype=torch.bool, device=d),
            truncated=Categorical(2, shape=(1,), dtype=torch.bool, device=d),
            shape=self.batch_size,
        )

    def _set_seed(self, seed):
        self.rng = random.Random(seed)

    # ---------- game helpers ----------
    def _place_food(self):
        """Put food on a random free cell. Returns False if the board is full."""
        free = [(r, c) for r in range(self.n) for c in range(self.n)
                if (r, c) not in self.snake]
        if not free:
            return False
        self.food = self.rng.choice(free)
        return True

    def _in_bounds(self, cell):
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n

    def _get_obs(self):
        s = self.n - 1  # normalizer
        hr, hc = self.snake[0]
        fr, fc = self.food
        danger = []
        for dr, dc in DIRS:
            cell = (hr + dr, hc + dc)
            danger.append(float(not self._in_bounds(cell) or cell in self.snake))
        direction = [float(i == self.direction) for i in range(4)]
        obs = [hr / s, hc / s, fr / s, fc / s,
               (fr - hr) / s, (fc - hc) / s] + danger + direction
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _flags(self, terminated, truncated):
        f = lambda v: torch.tensor([v], dtype=torch.bool, device=self.device)
        return {"done": f(terminated or truncated),
                "terminated": f(terminated),
                "truncated": f(truncated)}

    # ---------- EnvBase API ----------
    def _reset(self, tensordict=None, **kwargs):
        mid = self.n // 2
        self.snake = [(mid, mid)]               # list of cells, head first
        self.direction = self.rng.randrange(4)
        self.idle = 0                           # steps since last food
        self._place_food()
        out = TensorDict({"observation": self._get_obs(),
                          **self._flags(False, False)},
                         batch_size=self.batch_size)
        return out

    def _step(self, tensordict):
        action = int(tensordict["action"].item())

        # Ignore a 180-degree reversal (would instantly hit the neck)
        if (action + 2) % 4 != self.direction:
            self.direction = action

        dr, dc = DIRS[self.direction]
        hr, hc = self.snake[0]
        new_head = (hr + dr, hc + dc)

        ate = new_head == self.food
        # If we don't eat, the tail cell is vacated this tick, so it's safe.
        body = self.snake if ate else self.snake[:-1]

        reward, terminated, truncated = -0.01, False, False

        if not self._in_bounds(new_head) or new_head in body:
            reward, terminated = -1.0, True      # died
        else:
            self.snake.insert(0, new_head)        # move head forward
            if ate:
                reward = 1.0
                self.idle = 0
                if not self._place_food():        # board full -> win
                    terminated = True
            else:
                self.snake.pop()                  # drop tail
                self.idle += 1
                truncated = self.idle >= self.max_idle

        return TensorDict(
            {"observation": self._get_obs(),
             "reward": torch.tensor([reward], dtype=torch.float32,
                                    device=self.device),
             **self._flags(terminated, truncated)},
            batch_size=self.batch_size,
        )
