# -------------------------------------------
# Advantage Actor Critic
# -------------------------------------------


import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt


class ACAgent():
    """Implements the Advantage Actor Critic algorithm.
    """

    def __init__(self, gamma: float, alpha_actor: float, alpha_critic: float,
                 input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Initialize agent.

        Args:
            gamma (float): Discount factor.
            alpha (float): Learning rate of the actor network.
            alpha (float): Learning rate of the critic network.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        self.gamma = gamma
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.name = "Advantage Actor Critic"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor = Network(input_dim, output_dim, hidden_dims)
        self.actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), alpha_actor)
        self.critic = Network(input_dim, 1, hidden_dims)
        self.critic.to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), alpha_critic)

    def train(self, env, episodes: int, max_steps: int = 1000) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.
            max_steps (int): Maximum number of steps per episode. Defaults to 1000.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        results = {"episode": [], "score": []}

        # Initialize Plot
        fig, ax = plt.subplots()
        points, = ax.plot(0, 0, marker=".", linestyle="", markersize=3, alpha=0.3)
        new_point,  = ax.plot(0, 0, marker="o", linestyle="", markersize=3, color="black")
        line, = ax.plot(0, 0, color="red", label="SMA100")
        ax.legend(handles=[line])

        plt.title(self.name)
        plt.axhline(y=200, color="black", linestyle="dashed", alpha=0.2)
        plt.xlim(0, episodes)
        plt.ylim(-500, 500)
        plt.xlabel("Episode")
        plt.ylabel("Score")

        pbar = trange(episodes)
        for episode in pbar:
            state, _ = env.reset()
            terminated, truncated = False, False
            score = 0.0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.optimize(state, action, reward, next_state, terminated)
                state = next_state

                score += reward

                if terminated or truncated:
                    break

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
            points.set_data(results["episode"][:-1], results["score"][:-1])
            new_point.set_data(results["episode"][-1], results["score"][-1])
            line.set_data(results["episode"], sma)
            fig.canvas.draw()
            plt.pause(1e-5)

            # Update progress bar
            pbar.set_description(f"score={score:.2f}")

        plt.show()

        return results

    def optimize(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, terminated: bool) -> None:
        """Optimize actor and critic networks.
        """
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)

        # Compute log probability of action taken
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action).to(self.device))

        # Update actor
        td_target = reward + self.gamma * self.critic(next_state) * (not terminated)
        td_error = (td_target - self.critic(state))
        loss_actor = -1 * td_error * log_prob
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Update critic: We have to built the graph a second time since PyTorch allows only one backward pass.
        td_target = reward + self.gamma * self.critic(next_state) * (not terminated)
        td_error = (td_target - self.critic(state))
        loss_critic = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

    def play(self, env, episodes: int) -> dict:
        """Play a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
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
            int: The choosen action.
        """
        logits = self.actor(torch.from_numpy(state).to(self.device))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return int(action.item())

    def save_model(self, file_name: str) -> None:
        """Save model.

        Args:
            file_name (str): File name of the model. A common PyTorch convention is 
            using .pt file extension. 
        """
        path = os.path.join(os.getcwd(), "models", "actor_" + file_name)
        torch.save(self.actor.state_dict(), path)
        path = os.path.join(os.getcwd(), "models", "critic_" + file_name)
        torch.save(self.critic.state_dict(), path)

    def load_model(self, file_name: str) -> None:
        """Load model.

        Args:
            file_name (str): File name of the model.
        """
        path = os.path.join(os.getcwd(), "models", "actor_" + file_name)
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        path = os.path.join(os.getcwd(), "models", "critic_" + file_name)
        self.critic.load_state_dict(torch.load(path, map_location=self.device))


class Network(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Initialize model.

        Args:
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        super(Network, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns a prediction given an observation.

        Args:
            state (np.ndarray): Observation of the environment.

        Returns:
            torch.Tensor: Logits for a given oberservation.
        """
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)

        return x
