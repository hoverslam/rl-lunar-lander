# -------------------------------------------
# Neural Fitted Q Iteration
# -------------------------------------------


import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
import matplotlib.pyplot as plt


class NFQAgent():

    def __init__(self, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float,
                 alpha: float, input_dim: int, output_dim: int, hidden_dims: list[int],
                 epochs: int = 30) -> None:
        """An agent implemented by Neural Fitted Q Iteration.

        Args:
            gamma (float): Discount factor.
            epsilon_init (float): Start value of epsilon schedule.
            epsilon_min (float): Minimum value of epsilon schedule.
            epsilon_decay (float): Decay rate of epsilon values.
            alpha (float): Learning rate of the optimizer.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
            epochs (int): Number of training passes of the memory every episode. Defaults to 50.
        """
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.name = "Neural Fitted Q Iteration"
        self.model = FCQ(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.epochs = epochs

    def train(self, env, episodes: int, max_steps: int = 1000) -> None:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.
            max_steps (int): Maximum number of steps per episode. Defaults to 1000.
        """
        self.model.train()
        epsilons = self.decay_schedule(self.epsilon_init, self.epsilon_min, self.epsilon_decay, episodes)

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
            self.memory = []
            terminated, truncated = False, False
            score = 0

            for t in range(max_steps):
                action = self.act(state, epsilons[episode])
                next_state, reward, terminated, truncated, _ = env.step(action)

                self.memory.append((state, action, reward, next_state, terminated))
                state = next_state
                score += reward

                if terminated or truncated:
                    break

            self.optimize()

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

    def optimize(self) -> None:
        """Fit agent model.
        """
        states, actions, rewards, next_states, terminated = list(zip(*self.memory))
        states = np.array(states, dtype=np.float32)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.model.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        next_states = np.array(next_states, dtype=np.float32)
        terminated = torch.tensor(terminated, dtype=torch.int8, device=self.model.device)

        for _ in range(self.epochs):
            pred_qs = self.model(states).gather(1, actions).squeeze()
            max_qs = self.model(next_states).max(1)[0]
            target_qs = rewards + self.gamma * max_qs * (1 - terminated)

            self.optimizer.zero_grad()
            loss = self.criterion(pred_qs, target_qs)
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
        self.model.eval()
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

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose (optimal) action given an observation.

        Args:
            state (np.ndarray): An observation from the environment.
            epsilon (float, optional): Based on epsilon the agent takes random (high) or greedy (low) actions. 
            Defaults to 0.0.

        Returns:
            int: Action to take as an integer.
        """
        if torch.rand(1) < epsilon:
            return int(torch.randint(high=self.output_dim, size=(1,)).item())
        else:
            return self.model(state).detach().argmax().item()

    def decay_schedule(self, init_value: float, min_value: float, decay_ratio: float, max_steps: int) -> np.ndarray:
        """Compute exponentially decaying values (e.g. epsilons) for the complete training process in advance. 
        See: Morales (2020) Grooking DRL

        Args:
            init_value (float): An initial value.
            min_value (float): A minimum value.
            decay_ratio (float): Percentage of the max_steps to decay the values from initial to minimum.
            max_steps (int): The length of the schedule. This should be the number of training episodes.

        Returns:
            np.ndarray: An array of decaying values.
        """
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(start=-2, stop=0, num=decay_steps)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")

        return values

    def save_model(self, file_name: str) -> None:
        """Save model.

        Args:
            file_name (str): File name of the model. A common PyTorch convention is 
            using .pt file extension. 
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name: str) -> None:
        """Load model.

        Args:
            file_name (str): File name of the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.model.load_state_dict(torch.load(path, map_location=self.model.device))


class FCQ(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Initialize model.
        """
        super(FCQ, self).__init__()

        # Nice way to make the network architecture flexible. See: Morales (2020) Grooking DRL
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """Make a prediction given an observation.

        Args:
            state (np.ndarray): Observation of the environment.

        Returns:
            torch.Tensor: Q-values for a given observation.
        """
        x = torch.tensor(state, dtype=torch.float32, device=self.device)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)

        return x
