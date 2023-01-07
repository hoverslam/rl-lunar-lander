# -------------------------------------------
# Vanilla Policy Gradient / REINFORCE
# -------------------------------------------


import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt


class VPGAgent():

    def __init__(self, gamma: float, alpha: float, input_dim: int, output_dim: int,
                 hidden_dims: list[int]) -> None:
        """An agent using the Vanilla Policy Gradient algorithm.

        Args:
            gamma (float): Discount factor.
            alpha (float): Learning rate of the optimizer.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.name = "Vanilla Policy Gradient"
        self.policy = PolicyNet(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)

    def train(self, env, episodes: int, max_steps: int = 1000) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.
            max_steps (int): Maximum number of steps per episode. Defaults to 1000.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        self.policy.train()

        results = {"episode": [], "score": []}

        # Initialize Plot
        fig, ax = plt.subplots()
        point, = ax.plot(0, 0, marker="o", linestyle="", markersize=3, alpha=0.3)
        line, = ax.plot(0, -200, color="red")
        plt.xlim(0, episodes)
        plt.ylim(-500, 500)

        pbar = trange(episodes)
        for episode in pbar:
            state, _ = env.reset()
            terminated, truncated = False, False
            states = []
            actions = []
            rewards = []
            score = 0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                score += reward

                if terminated or truncated:
                    break

            discounted_returns = self.calculate_returns(rewards)
            self.optimize(states, actions, discounted_returns)

            # Update stats
            results["episode"].append(episode+1)
            results["score"].append(score)

            # Compute simple moving average
            sma = []
            window_size = 100
            for i in range(len(results["score"])):
                if i < window_size:
                    sma.append(sum(results["score"][:i+1]) / (i+1))
                else:
                    sma.append(sum(results["score"][i-window_size+1:i+1]) / window_size)

            # Update plot
            point.set_data(results["episode"], results["score"])
            line.set_data(results["episode"], sma)
            fig.canvas.draw()
            plt.pause(1e-5)

            # Update progress bar
            pbar.set_description(f"score={score:.2f}")

        plt.show()

        return results

    def calculate_returns(self, rewards: list[float]) -> np.ndarray:
        """Discount rewards for every step in an episode.

        Args:
            rewards (list[float]): List of rewards.

        Returns:
            np.ndarray: Discounted future rewards.
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        future_return = 0.0
        for t in reversed(range(T)):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return

        return returns

    def optimize(self, states: list[np.ndarray], actions: list[int], returns: np.ndarray) -> None:
        """Fit policy.

        Args:
            states (list[np.ndarray]): Observations from an environment.
            actions (list[int]): Actions taken given observations.
            returns (np.ndarray): Discounted future returns received during an episode. 
        """
        states = np.array(states, dtype=np.float32)
        actions = torch.tensor(actions, dtype=torch.int8, device=self.policy.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)

        mean = returns.mean()
        std = returns.std()
        normalized_returns = (returns - mean) / std.clamp_min(1e-10)

        logits = self.policy(states)
        pd = torch.distributions.Categorical(logits=logits)
        log_probs = pd.log_prob(actions)

        self.optimizer.zero_grad()
        loss = (-1 * log_probs * normalized_returns).sum()
        loss.backward()
        self.optimizer.step()

    def play(self, env, episodes: int) -> dict:
        """Play a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        self.policy.eval()
        results = {"episode": [], "score": []}

        for episode in range(episodes):
            state, _ = env.reset()
            terminated, truncated = False, False
            score = 0

            while (not terminated) and (not truncated):
                action = self.act(state)
                state, reward, terminated, truncated, _ = env.step(action)

                score += reward

            print("{}/{}: {:.2f}".format(episode+1, episodes, score))
            results["episode"].append(episode+1)
            results["score"].append(score)

        return results

    def act(self, state: np.ndarray) -> int:
        """Returns (optimal) action given an observation.

        Args:
            state (np.ndarray): An observation from an environment.

        Returns:
            int: Action to take as integer.
        """
        logits = self.policy(state)
        pd = torch.distributions.Categorical(logits=logits)
        action = int(pd.sample().item())

        return action

    def save_model(self, file_name: str) -> None:
        """Save model.

        Args:
            file_name (str): File name of the model. A common PyTorch convention is 
            using .pt file extension. 
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        torch.save(self.policy.state_dict(), path)

    def load_model(self, file_name: str) -> None:
        """Load model.

        Args:
            file_name (str): File name of the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.policy.load_state_dict(torch.load(path, map_location=self.policy.device))


class PolicyNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Initialize model.

        Args:
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        super(PolicyNet, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """Returns a prediction given an observation.

        Args:
            state (np.ndarray): Observation of the environment.

        Returns:
            torch.Tensor: Logits for a given oberservation.
        """
        x = torch.tensor(state, dtype=torch.float32, device=self.device)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)

        return x
