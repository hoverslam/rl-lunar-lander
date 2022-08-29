import gym
import sys
import typer

sys.path.append("modules")
from modules.nfq import NFQAgent
from modules.dqn import DQNAgent


app = typer.Typer()

agents = {"NFQ": NFQAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.2, epsilon_decay=0.9, 
                          alpha=0.001, input_dim=8, output_dim=4, hidden_dims=[128, 64]),
          "DQN": DQNAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.9, 
                          alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[128, 64])}

@app.command()
def train(algo: str, episodes: int) -> None:
    env = gym.make("LunarLander-v2", new_step_api=True)

    a = agents[algo]
    results = a.train(env, episodes)
    a.save_model("{}.pt".format(algo))
    
    a.plot_results(results, a.name)

    env.close()
    
@app.command()
def evaluate(algo: str, episodes: int, render: bool) -> None:
    if render:
        env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True)
    else:
        env = gym.make("LunarLander-v2", new_step_api=True)
        
    a = agents[algo]
    a.load_model("{}.pt".format(algo))
    results = a.play(env, episodes)
    
    average = sum(results["score"]) / len(results["score"])
    print("----------------------")     
    print("Average score: {:.2f}".format(average))
    print("----------------------")
    
    print("")
    input("Press ENTER to exit.")
    
    env.close()


if __name__ == "__main__":
    app()