# -------------------------------------------
# Deep Q-Network
# -------------------------------------------


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from base import Agent


class DQNAgent(Agent):
    
    def __init__(self, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float, 
                 alpha: float, input_dim: int, output_dim: int, hidden_dims: list[int],
                 memory_size: int = 10000, batch_size: int = 512) -> None:
        """An agent implemented with a Deep Q-Network.

        Args:
            gamma (float): Discount factor.
            epsilon_init (float): Start value of epsilon schedule.
            epsilon_min (float): Minimum value of epsilon schedule.
            epsilon_decay (float): Decay rate of epsilon values.
            alpha (float): Learning rate of the optimizer.
            input_dim (int): Number of input dimensions (i.e. state space)
            output_dim (int): Number of available actions (i.e. action space)
            hidden_dims (list[int]): A list with units per layer.
            memory_size (int): Size of replay memory. Defaults to 10000.
            batch_size (int): Number of samples drawn from the replay memory. Defaults to 512.
        """
        super().__init__(gamma, epsilon_init, epsilon_min, epsilon_decay, 
                         alpha, input_dim, output_dim, hidden_dims)

        self.name = "Deep Q-Network"
        self.model = DQN(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(max_size=memory_size)
        self.batch_size = batch_size
        
    def train(self, env, episodes):
        # TODO: Testing
        self.model.train()
        epsilons = self.decay_schedule(self.epsilon_init, self.epsilon_min, self.epsilon_decay, episodes)
        results = {"episode": [], "score": []}
        
        for episode in episodes:
            state = env.reset()
            terminated, truncated = False, False
            score = 0
            
            while (not terminated) and (not truncated):
                action = self.act(state, epsilons[episode])
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.memory.append((state, action, reward, next_state, terminated))
                state = next_state
                score += reward
                
                if len(self.memory) > (5 * batch_size):
                    self.optimize()
                    
            results["episode"].append(episode+1)
            results["score"].append(score)
            
        return results
        
    def play(self, env, episodes):
        # TODO: Implement
        pass
    
    def optimize(self):
        # TODO: Implement
        
        # 1.) sample batch from memory
        # 2.) make predictions
        # 3.) set Q-Learning targets
        # 4.) fit on MSE
        
        pass

   
class DQN(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]):
        super(DQN, self).__init__()        
        # TODO: Implement
        pass
    
    def forward(self, state):
        # TODO: Implement
        pass
    
    
class ReplayMemory():
    # TODO: Write documentation
    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0
        self.index = 0
        self.rng = np.random.default_rng()
        
        self.states = [None] * max_size
        self.actions = [None] * max_size
        self.rewards = [None] * max_size
        self.next_states = [None] * max_size
        self.terminated = [None] * max_size
    
    def __len__(self):
        return self.size
    
    def append(self, tup):
        self.states[self.index] = tup[0]
        self.actions[self.index] = tup[1]
        self.rewards[self.index] = tup[2]
        self.next_states[self.index] = tup[3]
        self.terminated[self.index] = tup[4]
        
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        
    def sample(self, batch_size):
        idx = self.rng.choice(self.size, batch_size, replace=False)
        
        return (
            [self.states[i] for i in idx],
            [self.actions[i] for i in idx],
            [self.rewards[i] for i in idx],
            [self.next_states[i] for i in idx],
            [self.terminated[i] for i in idx]
        )