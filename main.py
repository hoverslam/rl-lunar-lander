import gym
import sys

sys.path.append("modules")
from nfq import NFQAgent


agents = {"NFQ": NFQAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.9, 
                          alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1028, 512])}


def train(episodes: int) -> None:
    env = gym.make("LunarLander-v2", new_step_api=True)

    a = agents["NFQ"]
    results = a.train(env, episodes, 50)
    a.plot_results(results, a.name)
    a.save_model("nfq.pt")
    
    env.close()
    
def show(episodes: int) -> None:
    env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True)
        
    a = agents["NFQ"]
    a.load_model("nfq.pt")
    a.play(env, episodes)
    
    env.close()
    

if __name__ == "__main__":
    train(1000)
    show(10)