import logging

from tqdm import tqdm
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from gym import spaces
from cube_utils import apply_move, scramble_cube, get_solved_state

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


class RubiksCubeEnv(gym.Env):
    def __init__(self, num_moves=1, max_steps=100):
        super(RubiksCubeEnv, self).__init__()
        self.num_moves = num_moves
        self.max_steps = max_steps
        self.actions = [
            "-",
            "U",
            "U'",
            "D",
            "D'",
            "F",
            "F'",
            "B",
            "B'",
            "L",
            "L'",
            "R",
            "R'",
        ]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(6, 3, 3), dtype=np.int32
        )
        self.solved_state = get_solved_state()
        self.state = None

    def reset(self):
        self.num_steps = 0
        # Randomly select the between 1 and self.num_moves
        current_num_moves = np.random.randint(1, self.num_moves + 1)
        self.state, moves = scramble_cube(n_moves=current_num_moves)
        return self.state.copy(), moves

    def step(self, action):
        self.num_steps += 1
        move = self.actions[action]
        self.state = apply_move(self.state, move)

        cube_solved = np.array_equal(self.state, self.solved_state)
        done = cube_solved or self.num_steps >= self.max_steps
        reward = 1 if cube_solved else 0

        return self.state.copy(), reward, done, {}

    def render(self, mode="human"):
        print(self.state)


class CubeDQN(nn.Module):
    def __init__(self, num_actions):
        super(CubeDQN, self).__init__()
        self.num_actions = num_actions

        # Calculate input size:
        # 6 faces * 3x3 stickers * 6 colors (one-hot encoded)
        # 6 * 9 * 6 = 324 input features
        input_size = 324

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def forward(self, x):
        # x comes in as shape (Batch_Size, 6, 3, 3) with integers 0-5

        # 1. Ensure input is Long (integer) type for one_hot encoding
        x = x.long()

        # 2. One-Hot Encode
        # Transforms each number into a vector of 6 zeros/ones.
        # Example: 0 -> [1, 0, 0, 0, 0, 0], 5 -> [0, 0, 0, 0, 0, 1]
        # New shape: (Batch_Size, 6, 3, 3, 6)
        x_one_hot: torch.Tensor = F.one_hot(x, num_classes=6)

        # 3. Flatten
        # Flatten everything into a single long vector per sample
        # New shape: (Batch_Size, 324)
        x_flat = x_one_hot.view(x.size(0), -1).float()
        return self.model(x_flat)


if __name__ == "__main__":

    # Use Apple Silicon GPU (MPS) when available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = RubiksCubeEnv(num_moves=12, max_steps=20)

    # 1. Create the Policy Network (the one that learns)
    model = CubeDQN(num_actions=len(env.actions)).to(device)

    # 2. Create the Target Network (the one that predicts the future)
    target_net: CubeDQN = CubeDQN(num_actions=len(env.actions)).to(device)
    target_net.load_state_dict(model.state_dict())  # Copy the weights
    target_net.eval()  # Set to evaluation mode

    # 3. Create the Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.SmoothL1Loss()

    num_episodes = 4000
    gamma = 0.9
    epsilon = 0.9
    epsilon_decay = 0.995
    memory = deque(maxlen=10_000)
    batch_size = 64

    episode_rewards = []
    progress_bar = tqdm(range(num_episodes), desc="Training", unit="episode")
    for episode in progress_bar:
        state, moves = env.reset()
        logging.debug(f"Episode {episode + 1}")
        logging.debug(f"Scrambled State: {moves}")
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)

        done = False
        agent_moves: list[str] = []
        step_rewards = []
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.inference_mode():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()

            agent_moves.append(env.actions[action])
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.tensor(next_state).unsqueeze(0).to(device)

            memory.append(
                (
                    state_tensor.detach().cpu(),
                    action,
                    reward,
                    next_state_tensor.detach().cpu(),
                    done,
                )
            )
            step_rewards.append(reward)
            state_tensor = next_state_tensor

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).to(device)
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.cat(next_states).to(device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                # max_next_q = model(next_states).max(1)[0].detach()

                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + (gamma * max_next_q * (1 - dones))

                loss = criterion(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # Update the Target Network
        if episode % 20 == 0:
            target_net.load_state_dict(model.state_dict())

        logging.debug(f"Agent Moves: {agent_moves}")
        episode_reward = sum(step_rewards)
        episode_rewards.append(episode_reward)

        if episode_reward > 0:
            logging.debug(f"Cube Solved")
        else:
            logging.debug(f"Cube Not Solved")

        epsilon *= epsilon_decay
        avg_loss = loss.item() if "loss" in locals() else 0.0
        progress_bar.set_postfix(
            {
                "Loss": f"{avg_loss:.4f}",
                "Epsilon": f"{epsilon:.2f}",
                "Total Rewards": f"{sum(episode_rewards):.0f}",
            }
        )
    logging.info("Training completed!")
