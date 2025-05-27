import collections
import random
import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class AgentDQN(Agent):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.hidden_size = getattr(args, "hidden_size", 128)
        self.lr = getattr(args, "lr", 1e-3)
        self.gamma = getattr(args, "gamma", 0.99)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 500
        self.batch_size = 64
        self.buffer_size = 10000
        self.target_update_freq = 100
        self.device = torch.device(
            "cuda"
            if (getattr(args, "use_cuda", False) and torch.cuda.is_available())
            else "cpu"
        )
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.qnet = QNetwork(self.input_size, self.hidden_size, self.output_size).to(
            self.device
        )
        self.target_qnet = QNetwork(
            self.input_size, self.hidden_size, self.output_size
        ).to(self.device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter()
        self.steps_done = 0
        self.epsilon = self.epsilon_start

    def make_action(self, observation, test=True):
        obs = (
            torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        if test:
            with torch.no_grad():
                return self.qnet(obs).argmax(1).item()
        else:
            self.epsilon = self.epsilon_end + (
                self.epsilon_start - self.epsilon_end
            ) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            if random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                with torch.no_grad():
                    return self.qnet(obs).argmax(1).item()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(
            self.device
        )
        action_batch = (
            torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
        )
        reward_batch = (
            torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(
            self.device
        )
        done_batch = (
            torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        q_values = self.qnet(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_qnet(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        best_avg_reward = -float("inf")
        reward_history = []
        all_steps = 0
        max_episodes = 500
        for episode in range(max_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # 兼容新gym
                state = state[0]
            episode_reward = 0
            done = False
            while not done:
                action = self.make_action(state, test=False)
                result = self.env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                self.replay_buffer.push(
                    (state, action, reward, next_state, float(done))
                )
                state = next_state
                episode_reward += reward
                all_steps += 1
                self.train()
                if all_steps % self.target_update_freq == 0:
                    self.target_qnet.load_state_dict(self.qnet.state_dict())
            reward_history.append(episode_reward)
            avg_reward = np.mean(reward_history[-20:])
            print(
                f"Episode {episode}, Reward: {episode_reward}, Avg_Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}"
            )
            self.writer.add_scalar("Reward", episode_reward, episode)
            self.writer.add_scalar("AvgReward", avg_reward, episode)
            # Early stop if solved
            if avg_reward >= 195:  # CartPole-v0 评测标准
                print(f"Solved at episode {episode}!")
                break
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(self.qnet.state_dict(), f"best_dqn.pth")
        self.writer.close()
