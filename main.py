import gym
from agents import NFQ


def train(episodes: int) -> None:
    env = gym.make("LunarLander-v2", new_step_api=True)

    agent = NFQ(env, gamma=0.99, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1028, 512])
    results = agent.train(episodes, 50)
    agent.plot_results(results, "Neural Fitted Q Iteration")
    agent.save_model("nfq.pt")
    
    env.close()
    
def play(episodes: int, render: bool = True) -> None:
    if render:
        env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True)
    else:
        env = gym.make("LunarLander-v2", new_step_api=True)
        
    agent = NFQ(env, gamma=0.99, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1028, 512])
    agent.load_model("nfq.pt")
    agent.play(episodes)
    
    env.close()
    

if __name__ == "__main__":
    train(1000)
    play(10)