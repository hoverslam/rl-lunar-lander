# Lunar Lander

The *Lunar Lander* environment is a rocket trajectory optimization problem. The goal is to touch down at the landing pad as close as possible. The rocket starts at the top center with a random initial force applied to its center of mass.

There are four discrete action: do nothing, fire left engine, fire main engine, and fire right engine.

Each observation is an 8-dimensional vector containing: the lander position in *x* & *y*, its linear velocity in *x* & *y*, its angle, its angular velocity, and two boolean flags indicating whether each leg has contact with the ground.

Positive rewards are received for a landing (100-140, depending on the position) with +100 if the lander comes to a rest. Firing the engines gives a tiny (-0.03) and crashing a big (-100) negative reward. The problem is considered solved by reaching 200 points.

The following RL algorithms were implemented:
- Neural Fitted Q Iteration
- Deep Q-Network
- Vanilla Policy Gradient / REINFORCE

## How to

Install dependencies with &nbsp;&nbsp;&nbsp;&nbsp; `pip install -r requirements.txt`.

Run &nbsp;&nbsp;&nbsp;&nbsp; `main.py train <agent> <episodes>` &nbsp;&nbsp;&nbsp;&nbsp; to train an agent.

Run &nbsp;&nbsp;&nbsp;&nbsp; `main.py evaluate <agent> <episodes> <render>` &nbsp;&nbsp;&nbsp;&nbsp; to evaluate a pre-trained agent.

`<agent> (string)` &nbsp;&nbsp; NFQ, DQN or VPG

`<episodes> (int)` &nbsp;&nbsp; Number of episodes

`<render> (bool)` &nbsp;&nbsp;&nbsp;&nbsp; Display episodes on screen

## Neural Fitted Q Iteration (NFQ)

| Training                                                    | After 3000 episodes                                |
|:-----------------------------------------------------------:|:--------------------------------------------------:|
| <img src="../media/NFQ_training.png?raw=true" height="300"> | <img src="../media/NFQ.gif?raw=true" height="300"> |

Reference: [M. Riedmiller (2005) Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method](https://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf)

## Deep Q-Network (DQN)

| Training                                                    | After 1000 episodes                                |
|:-----------------------------------------------------------:|:--------------------------------------------------:|
| <img src="../media/DQN_training.png?raw=true" height="300"> | <img src="../media/DQN.gif?raw=true" height="300"> |

Reference: [V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller (2013) Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

## Vanilla Policy Gradient / REINFORCE (VPG)

| Training                                                    | After 5000 episodes                                |
|:-----------------------------------------------------------:|:--------------------------------------------------:|
| <img src="../media/VPG_training.png?raw=true" height="300"> | <img src="../media/VPG.gif?raw=true" height="300"> |

Reference: [R. Sutton, and A. Barto (2018) Reinforcement Learning: An Introduction, p. 328](http://incompleteideas.net/book/the-book.html)

## Comparison

The score is the average return over 100 episodes on the trained agent.

|                           |  Score |
|---------------------------|:------:|
| Neural Fitted Q Iteration | -39.08 |
| Deep Q-Network            | 259.08 |
| Vanilla Policy Gradient   | 166.73 |

## Dependencies

- Python v3.10.4
- Gym v0.26.0
- Matplotlib v3.5.3
- Numpy v1.23.3
- Pandas v1.4.4
- PyTorch v1.12.1
- Tqdm v4.64.1
- Typer v0.6.1