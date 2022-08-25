import gym
import sys

sys.path.append("modules")
from modules.nfq import NFQAgent
from modules.dqn import DQNAgent


agents = {"NFQ": NFQAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.01, epsilon_decay=0.9, 
                          alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1024, 512]),
          "DQN": DQNAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.01, epsilon_decay=0.9, 
                          alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1024, 512])}


def train(episodes: int, algo: str) -> None:
    env = gym.make("LunarLander-v2", new_step_api=True)

    a = agents[algo]
    results = a.train(env, episodes)
    a.plot_results(results, a.name)
    a.save_model("{}.pt".format(algo))
    
    env.close()
    
def show(episodes: int, algo: str) -> None:
    env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True)
        
    a = agents[algo]
    a.load_model("{}.pt".format(algo))
    a.play(env, episodes)
    
    env.close()
    

if __name__ == "__main__":
    train(1000, "NFQ")
    show(10, "NFQ")