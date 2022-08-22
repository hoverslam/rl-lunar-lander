import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod


class Agent(ABC):
    
    @abstractmethod
    def __init__(self, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float, 
                 alpha: float, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Base class for all agents.

        Args:
            env (_type_): An OpenAI gym environment.
            gamma (float): Discount factor.
            epsilon_init (float): Start value of epsilon schedule.
            epsilon_min (float): Minimum value of epsilon schedule.
            epsilon_decay (float): Decay rate of epsilon values.
            alpha (float): Learning rate of the optimizer.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        self.gamma = gamma        
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        self.rng = np.random.default_rng()
        
    @abstractmethod
    def train(self, episodes: int) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            episodes (int): Number of episodes.

        Raises:
            NotImplementedError: Implement in subclass.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        raise NotImplementedError
    
    @abstractmethod
    def play(self, episodes: int) -> None:
        """Play a given number of episodes.

        Args:
            episodes (int): Number of episodes.

        Raises:
            NotImplementedError: Implement in subclass.
        """
        raise NotImplementedError        
    
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose (optimal) action given an observation.

        Args:
            state (np.ndarray): An observation from the environment.
            epsilon (float, optional): Based on epsilon the agent takes random (high) or greedy (low) actions. 
            Defaults to 0.0.

        Returns:
            int: Action to take as an integer.
        """
        if self.rng.random() < epsilon:
            return self.rng.choice(range(self.output_dim))
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
        torch.save(self.model, path)
    
    def load_model(self, file_name: str) -> None:
        """Load model.

        Args:
            file_name (str): File name of the model.
        """
        path = os.path.join(os.getcwd(), "models", file_name)
        self.model = torch.load(path)
    
    @staticmethod
    def plot_results(results: dict, title: str) -> None:
        """Plot results of all episodes.

        Args:
            results (dict): A dictionary containing the score of each episode.
            title (str): Plot title.
        """
        results = pd.DataFrame(results)
        results["sma100"] = results["score"].rolling(100).mean()
        _, ax = plt.subplots()
        ax.plot(results["episode"], results["sma100"], c="red", label="SMA100")
        ax.scatter(results["episode"], results["score"], s=2, alpha=0.2)
        ax.legend()
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        plt.title(title)        
        plt.show()


# -------------------------------------------
# Neural Fitted Q Iteration
# -------------------------------------------

class NFQ(Agent):
    
    def __init__(self, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float, 
                 alpha: float, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """An agent using the Neural Fitted Q Iteration.

        Args:
            gamma (float): Discount factor.
            epsilon_init (float): Start value of epsilon schedule.
            epsilon_min (float): Minimum value of epsilon schedule.
            epsilon_decay (float): Decay rate of epsilon values.
            alpha (float): Learning rate of the optimizer.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
        """
        super().__init__(gamma, epsilon_init, epsilon_min, epsilon_decay, 
                         alpha, input_dim, output_dim, hidden_dims)

        self.name = "Neural Fitted Q Iteration"
        self.model = FCQ(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        
    def train(self, env, episodes: int, epochs: int) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.
            epochs (int): Number of epochs per fitting.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        self.model.train()
        epsilons = self.decay_schedule(self.epsilon_init, self.epsilon_min, self.epsilon_decay, episodes)
        results = {"episode": [], "score": []}
        
        for episode in tqdm(range(episodes)):
            state = env.reset()
            self.memory = []
            terminated, truncated = False, False
            score = 0
            
            while (not terminated) and (not truncated):
                action = self.act(state, epsilons[episode])
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.memory.append((state, action, reward, next_state, terminated))                
                state = next_state
                score += reward     
            
            self.optimize(epochs)
            
            results["episode"].append(episode+1)
            results["score"].append(score)
            
        return results
    
    def play(self, env, episodes: int) -> None:
        """Play a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        self.model.eval()
        
        for episode in range(episodes):
            state = env.reset()
            terminated, truncated = False, False
            score = 0
            
            while (not terminated) and (not truncated):
                action = self.act(state)
                state, reward, terminated, truncated, _ = env.step(action)
                                
                score += reward  
        
            print("{}/{}: Score = {:.2f}".format(episode+1, episodes, score))              
            
    def optimize(self, epochs: int) -> None:
        """Fit the agent memory to the model.

        Args:
            epochs (int): Number of epochs.
        """
        states, actions, rewards, next_states, terminated = list(zip(*self.memory))
        states = np.array(states, dtype=np.float32)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.model.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        next_states = np.array(next_states, dtype=np.float32)
        terminated = torch.tensor(terminated, dtype=torch.int8, device=self.model.device)

        for _ in range(epochs):
            pred_qs = self.model(states).gather(1, actions).squeeze()
            max_qs = self.model(next_states).detach().max(1)[0]
            target_qs = rewards + self.gamma * max_qs * (1 - terminated)

            self.optimizer.zero_grad()
            loss = self.criterion(pred_qs, target_qs)
            loss.backward()
            self.optimizer.step()
            

class FCQ(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        super(FCQ, self).__init__()
        """Initialize the model.
        """        
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


# -------------------------------------------
# Deep Q-Network
# -------------------------------------------

# TODO