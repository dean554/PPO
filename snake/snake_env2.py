import pygame
import sys
import random
import torch
from tensordict import TensorDict

pygame.init()

WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 10

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake - Pure AI Policy Mode")
clock = pygame.time.Clock()

def reset_game():
    global snake, direction, food, steps
    snake = [[5, 5], [4, 5], [3, 5]]
    direction = 1  # 0: Up, 1: Right, 2: Down, 3: Left
    food = [random.randint(0, 19), random.randint(0, 19)]
    steps = 0

reset_game()

# Helper: Convert action index (0-3) to delta velocities (dx, dy)
def action_to_dx_dy(action):
    if action == 0: return 0, -1   # Up
    if action == 1: return 1, 0   # Right
    if action == 2: return 0, 1   # Down
    if action == 3: return -1, 0  # Left
    return 1, 0

# Helper: Extract observation vector matching your RL environment specs
def get_observation(snake, food, grid_size=20):
    hx, hy = snake[0]
    fx, fy = food
    
    def is_danger(pos):
        x, y = pos
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size: return 1.0
        if [x, y] in snake: return 1.0
        return 0.0

    return torch.tensor([
        float(hx) / grid_size,
        float(hy) / grid_size,
        float(fx) / grid_size,
        float(fy) / grid_size,
        is_danger([hx, hy - 1]),
        is_danger([hx + 1, hy]),
        is_danger([hx, hy + 1]),
        is_danger([hx - 1, hy])
    ], dtype=torch.float32)

# Load trained policy module here (e.g., policy_module = torch.load("snake_policy.pt"))
# For demonstration, we'll keep a mock placeholder or can plug in trained model.
policy_module = None 

while True:
    # 1. Handle window close events only (no keyboard controls)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 2. Get action exclusively from the AI Policy Network
    with torch.no_grad():
        obs = get_observation(snake, food)
        
        if policy_module is not None:
            # Format observation into a TensorDict expected by TorchRL policies
            tensordict_in = TensorDict({"observation": obs}, batch_size=[])
            action_dict = policy_module(tensordict_in)
            chosen_action = action_dict["action"].item()
        else:
            # Stub/Random fallback if no model file is loaded yet
            chosen_action = random.randint(0, 3)

    # 3. Prevent 180-degree immediate reverse turns
    is_opposite = (
        (direction == 0 and chosen_action == 2) or
        (direction == 2 and chosen_action == 0) or
        (direction == 1 and chosen_action == 3) or
        (direction == 3 and chosen_action == 1)
    )
    if not is_opposite:
        direction = chosen_action

    # 4. Convert action code to movement offsets and step forward
    dx, dy = action_to_dx_dy(direction)
    new_head = [snake[0][0] + dx, snake[0][1] + dy]
    steps += 1

    # 5. Game over / collision handling
    hit_wall = not (0 <= new_head[0] < WIDTH // GRID_SIZE and 0 <= new_head[1] < HEIGHT // GRID_SIZE)
    hit_self = new_head in snake
    if hit_wall or hit_self or steps > 300:
        reset_game()
        continue

    # 6. Food & Growth logic
    if new_head == food:
        snake.insert(0, new_head)
        food = [random.randint(0, 19), random.randint(0, 19)]
    else:
        snake.insert(0, new_head)
        snake.pop()

    # 7. Render graphics
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 0, 0), (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    for segment in snake:
        pygame.draw.rect(screen, (0, 255, 0), (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()
    clock.tick(FPS)