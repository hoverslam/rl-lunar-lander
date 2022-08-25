# -------------------------------------------
# Neural Fitted Q Iteration
# -------------------------------------------


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from base import Agent


class NFQAgent(Agent):
    
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
        super().__init__(gamma, epsilon_init, epsilon_min, epsilon_decay, 
                         alpha, input_dim, output_dim, hidden_dims)

        self.name = "Neural Fitted Q Iteration"
        self.model = FCQ(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        
    def train(self, env, episodes: int) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.

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
            
            self.optimize()
            
            results["episode"].append(episode+1)
            results["score"].append(score)
            
        return results              
            
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
            max_qs = self.model(next_states).detach().max(1)[0]
            target_qs = rewards + self.gamma * max_qs * (1 - terminated)

            self.optimizer.zero_grad()
            loss = self.criterion(pred_qs, target_qs)
            loss.backward()
            self.optimizer.step()
            

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