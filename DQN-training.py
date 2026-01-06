import logging

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    num_moves: int = 14
    max_steps: int = 20

    num_episodes: int = 4000
    gamma: float = 0.9

    epsilon_start: float = 0.9
    epsilon_decay: float = 0.995

    learning_rate: float = 1e-4
    memory_size: int = 10_000
    batch_size: int = 64

    target_update_every: int = 20
    grad_clip_norm: float = 1.0

    seed: int | None = None


class Runner:
    def __init__(self, config: TrainingConfig):
        self.config = config

        if config.seed is not None:
            self._set_seed(config.seed)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        logger.info(f"Using device: {self.device}")

        self.env = RubiksCubeEnv(num_moves=config.num_moves, max_steps=config.max_steps)

        self.policy_net = CubeDQN(num_actions=len(self.env.actions)).to(self.device)
        self.target_net = CubeDQN(num_actions=len(self.env.actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

        self.memory: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(
            maxlen=config.memory_size
        )

        self.epsilon = config.epsilon_start
        self.episode_rewards: list[float] = []

    def run(self) -> None:
        progress_bar = tqdm(
            range(self.config.num_episodes), desc="Training", unit="episode"
        )
        for episode in progress_bar:
            state, moves = self.env.reset()
            logger.debug("Episode %d", episode + 1)
            logger.debug("Scrambled State: %s", moves)

            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            done = False

            agent_moves: list[str] = []
            step_rewards: list[float] = []
            last_loss: float | None = None

            while not done:
                action = self._select_action(state_tensor)
                agent_moves.append(self.env.actions[action])

                next_state, reward, done, _ = self.env.step(action)
                next_state_tensor = (
                    torch.tensor(next_state).unsqueeze(0).to(self.device)
                )

                self._remember(
                    state_tensor=state_tensor,
                    action=action,
                    reward=reward,
                    next_state_tensor=next_state_tensor,
                    done=done,
                )
                step_rewards.append(reward)
                state_tensor = next_state_tensor

                loss_value = self._optimize_step()
                last_loss = loss_value

            if episode % self.config.target_update_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            logger.debug("Agent Moves: %s", agent_moves)

            episode_reward = sum(step_rewards)
            self.episode_rewards.append(episode_reward)

            if episode_reward > 0:
                logger.debug("Cube Solved")
            else:
                logger.debug("Cube Not Solved")

            self.epsilon *= self.config.epsilon_decay

            progress_bar.set_postfix(
                {
                    "Loss": f"{(last_loss or 0.0):.4f}",
                    "Epsilon": f"{self.epsilon:.2f}",
                    "Total Rewards": f"{sum(self.episode_rewards):.0f}",
                }
            )

        logger.info("Training completed!")

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_action(self, state_tensor: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        with torch.inference_mode():
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()

    def _remember(
        self,
        *,
        state_tensor: torch.Tensor,
        action: int,
        reward,
        next_state_tensor: torch.Tensor,
        done: bool,
    ) -> None:
        self.memory.append(
            (
                state_tensor.detach().cpu(),
                action,
                reward,
                next_state_tensor.detach().cpu(),
                done,
            )
        )

    def _optimize_step(self) -> torch.Tensor:
        if len(self.memory) < self.config.batch_size:
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.cat(states).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.cat(next_states).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = (
            self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        )
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
        target_q = rewards_t + (self.config.gamma * max_next_q * (1 - dones_t))

        loss: torch.Tensor = self.criterion(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=self.config.grad_clip_norm
        )
        self.optimizer.step()

        return loss


if __name__ == "__main__":
    Runner(TrainingConfig()).run()
