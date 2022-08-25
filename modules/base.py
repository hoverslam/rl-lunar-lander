import os
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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