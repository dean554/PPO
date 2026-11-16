import os
import time

import torch
import torch.nn as nn
from torchrl.envs.utils import step_mdp

from snake_env import SnakeEnv

# ---------- settings ----------
CHECKPOINT = "snake_policy.pt"
EPISODES = 5        # how many games to watch
DELAY = 0.12        # seconds between frames
SAMPLE = False      # False = always pick the best move, True = sample from the policy

# ---------- load checkpoint ----------
ckpt = torch.load(CHECKPOINT, map_location="cpu")
num_cells = ckpt["num_cells"]

env = SnakeEnv(grid_size=ckpt["grid_size"], device="cpu")

import pygame

CELL = 20
pygame.init()
screen = pygame.display.set_mode((env.n * CELL, env.n * CELL))
pygame.display.set_caption("Snake (PPO)")
clock = pygame.time.Clock()


def draw():
    screen.fill((20, 20, 20))
    fr, fc = env.food
    pygame.draw.rect(screen, (220, 60, 60), (fc * CELL, fr * CELL, CELL, CELL))
    for i, (r, c) in enumerate(env.snake):
        color = (120, 255, 120) if i == 0 else (40, 170, 40)    # head is brighter
        pygame.draw.rect(screen, color, (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2))
    pygame.display.flip()

# This must have the same layers as the actor in ppo2.py (minus the TorchRL wrappers)
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells), nn.Tanh(),
    nn.LazyLinear(num_cells), nn.Tanh(),
    nn.LazyLinear(num_cells), nn.Tanh(),
    nn.LazyLinear(ckpt["n_actions"]),
)

# LazyLinear doesn't know its input size until it sees data, so run one
# dummy forward pass to create the weights, then load the trained values in.
actor_net(env.reset()["observation"])
actor_net.load_state_dict(ckpt["actor_state_dict"])
actor_net.eval()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ---------- play ----------
for ep in range(EPISODES):
    td = env.reset()
    total_reward, steps = 0.0, 0

    while True:
        with torch.no_grad():
            logits = actor_net(td["observation"])

        if SAMPLE:
            action = torch.distributions.Categorical(logits=logits).sample()
        else:
            action = logits.argmax()

        td["action"] = action
        td = env.step(td)               # result lives under td["next", ...]

        total_reward += td["next", "reward"].item()
        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        draw()
        clock.tick(80)       # frames per second; lower is slower
        if td["next", "done"].item():
            break
        td = step_mdp(td)               # move "next" into the current slot

    print(f"Game over: {len(env.snake) - 1} food in {steps} steps")
    time.sleep(1.5)