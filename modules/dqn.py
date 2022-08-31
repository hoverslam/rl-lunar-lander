# -------------------------------------------
# Deep Q-Network
# -------------------------------------------


import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class DQNAgent():
    
    def __init__(self, gamma: float, epsilon_init: float, epsilon_min: float, epsilon_decay: float, 
                 alpha: float, input_dim: int, output_dim: int, hidden_dims: list[int],
                 memory_size: int = 50000, batch_size: int = 128, target_net_frequency: int = 5) -> None:
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
            target_net_frequency (int): Update frequency of the target network in episodes. Defaults to 1.
        """        
        self.gamma = gamma        
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.name = "Deep Q-Network"
        self.model = DQN(input_dim, output_dim, hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(max_size=memory_size)
        self.batch_size = batch_size
        self.target_net_frequency = target_net_frequency
        
        # Target network
        self.target_net = DQN(input_dim, output_dim, hidden_dims)
        self.target_net.eval()
        
    def train(self, env, episodes: int, max_steps: int = 1000) -> dict:
        """Train the agent on a given number of episodes.

        Args:
            env(_type_): An OpenAI gym environment.
            episodes (int): Number of episodes.
            max_steps (int): Maximum number of steps per episode. Defaults to 1000.

        Returns:
            dict: A dictionary containing the score of each episode.
        """
        self.model.train()
        epsilons = self.decay_schedule(self.epsilon_init, self.epsilon_min, self.epsilon_decay, episodes)
        results = {"episode": [], "score": []}
        
        for episode in tqdm(range(episodes)):
            state = env.reset()
            terminated, truncated = False, False
            score = 0
            
            for t in range(max_steps):
                action = self.act(state, epsilons[episode])
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.memory.append((state, action, reward, next_state, terminated))
                state = next_state
                score += reward

                # Only start training when replay memory is big enough
                if len(self.memory) > (10 * self.batch_size):
                    self.optimize()
  
                if truncated or terminated:
                    break
                
            if episode % self.target_net_frequency == 0:
                self.target_net.load_state_dict(self.model.state_dict())
                    
            results["episode"].append(episode+1)
            results["score"].append(score)
            
        return results
    
    def optimize(self) -> None:
        """Fit agent model.
        """
        states, actions, rewards, next_states, terminated = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.model.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.model.device)
        terminated = torch.tensor(terminated, dtype=torch.int8, device=self.model.device)

        pred_qs = self.model(states).gather(1, actions).squeeze()
        max_qs = self.target_net(next_states).detach().max(1)[0]
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
            state = env.reset()
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
            return torch.randint(high=self.output_dim, size=(1,)).item()           
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

  
class DQN(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        """Initialize model.
        """  
        super(DQN, self).__init__()        
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
    
    def forward(self, state) -> torch.Tensor:
        """Make a prediction given an observation.

        Args:
            state (np.ndarray): Observation of the environment.

        Returns:
            torch.Tensor: Q-values for a given observation.
        """
        if not isinstance(state, torch.Tensor):
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            x = state
            
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        
        return x
    
    
class ReplayMemory():
    
    def __init__(self, max_size: int) -> None:
        """Initialize memory.

        Args:
            max_size (int): Maximum size of memory.
        """
        self.max_size = max_size
        self.size = 0
        self.index = 0
        self.rng = np.random.default_rng()
        
        self.states = [None] * max_size
        self.actions = [None] * max_size
        self.rewards = [None] * max_size
        self.next_states = [None] * max_size
        self.terminated = [None] * max_size

    def __len__(self) -> int:
        """Return the number of items in memory.

        Returns:
            int: Size of memory.
        """
        return self.size
    
    def append(self, experience: tuple) -> None:
        """Add experience to memory.

        Args:
            experience (tuple): An experience containing state, action, reward, next_state, and
            a termination flag.
        """
        self.states[self.index] = experience[0]
        self.actions[self.index] = experience[1]
        self.rewards[self.index] = experience[2]
        self.next_states[self.index] = experience[3]
        self.terminated[self.index] = experience[4]
        
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        
    def sample(self, batch_size: int) -> list:
        """Return a sample of experiences.

        Args:
            batch_size (int): Number of experiences sampled.

        Returns:
            list: A number of experiences given by a batch size.
        """
        idx = self.rng.choice(self.size, batch_size, replace=False)
        
        return (
            np.stack([self.states[i] for i in idx]),
            np.array([self.actions[i] for i in idx]),
            np.array([self.rewards[i] for i in idx]),
            np.stack([self.next_states[i] for i in idx]),
            np.array([self.terminated[i] for i in idx])
        )