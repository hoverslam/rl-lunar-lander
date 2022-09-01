# Lunar Lander

<img src="https://www.gymlibrary.dev/_images/lunar_lander.gif" alt="Lunar Lander Demo" width="320">

The *Lunar Lander* environment is a rocket trajectory optimization problem. The goal is to touch down at the landing pad as close as possible. The rocket starts at the top center with a random initial force applied to its center of mass.

There are four discrete action: do nothing, fire left engine, fire main engine, and fire right engine.

Each observation is an 8-dimensional vector containing: the lander position in *x* & *y*, its linear velocity in *x* & *y*, its angle, its angular velocity, and two boolean flags indicating whether each leg has contact with the ground.

Positive rewards are received for a landing (100-140, depending on the position) with +100 if the lander comes to a rest. Firing the engines gives a tiny (-0.03) and crashing a big (-100) negative reward. The problem is considered solved by reaching 200 points.

The following RL algorithms were implemented:
- Neural Fitted Q Iteration
- Deep Q-Network
- Vanilla Policy Gradient / REINFORCE

## How to

## Neural Fitted Q Iteration (NFQ)

## Deep Q-Network (DQN)

## Vanilla Policy Gradient / REINFORCE (VPG)

## Comparison
