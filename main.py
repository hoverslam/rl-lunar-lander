import gym
import typer
import pandas as pd
import matplotlib.pyplot as plt

from modules.nfq import NFQAgent
from modules.dqn import DQNAgent
from modules.vpg import VPGAgent


app = typer.Typer()

agents = {"NFQ": NFQAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.2, epsilon_decay=0.9,
                          alpha=0.001, input_dim=8, output_dim=4, hidden_dims=[128, 64]),
          "DQN": DQNAgent(gamma=0.99, epsilon_init=1.0, epsilon_min=0.1, epsilon_decay=0.9,
                          alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[256, 128]),
          "VPG": VPGAgent(gamma=0.999, alpha=0.0001, input_dim=8, output_dim=4, hidden_dims=[1024, 512])}


def plot_results(results: dict, title: str) -> None:
    results = pd.DataFrame(results)
    results["sma100"] = results["score"].rolling(100).mean()
    _, ax = plt.subplots()
    ax.plot(results["episode"], results["sma100"], c="red", label="SMA100")
    ax.scatter(results["episode"], results["score"], s=2, alpha=0.5)
    ax.legend()
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    plt.axhline(y=200, color="black", linestyle="dashed", alpha=0.2)
    plt.title(title)
    plt.show()


@app.command()
def train(algo: str, episodes: int) -> None:
    env = gym.make("LunarLander-v2")

    a = agents[algo]
    results = a.train(env, episodes)
    a.save_model("{}.pt".format(algo))

    plot_results(results, a.name)

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
