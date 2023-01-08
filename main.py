import gym
import typer

from modules.nfq import NFQAgent
from modules.dqn import DQNAgent
from modules.vpg import VPGAgent


app = typer.Typer()

agents = {"NFQ": NFQAgent(gamma=0.999, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.8,
                          alpha=0.0005, input_dim=8, output_dim=4, hidden_dims=[128, 64]),
          "DQN": DQNAgent(gamma=0.999, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.8, alpha=0.0005,
                          input_dim=8, output_dim=4, hidden_dims=[128, 64], batch_size=64),
          "VPG": VPGAgent(gamma=0.999, alpha=0.001, input_dim=8, output_dim=4, hidden_dims=[128, 64])}


@app.command()
def train(algo: str, episodes: int) -> None:
    env = gym.make("LunarLander-v2")

    a = agents[algo]
    a.train(env, episodes)
    a.save_model("{}.pt".format(algo))

    env.close()


@app.command()
def evaluate(algo: str, episodes: int, render: bool) -> None:
    if render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")

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
