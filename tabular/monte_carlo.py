"""Implements monte carlo methods."""

import numpy as np

from utils.mdp import BaseMDP
from utils.agent import Agent


def first_visit_prediction(
    env: BaseMDP,
    policy: Agent,
    max_steps: int,
    num_episodes: int
) -> np.ndarray: 
    """Computes action-values via first-visit monte carlo prediction."""
    n_states = env.n_states
    n_actions = env.n_actions

    # Init action-values
    action_values = np.zeros((n_states, n_actions))
    visit_counts = np.zeros_like(action_values)

    for episode in range(num_episodes):
        # Sample trajectory
        trajectory = env.sample_trajectory(policy, max_steps)

        visited = set()
        G = 0
        for (s, a, r, s_next) in trajectory[::-1]:
            # Update return
            G = env.discount_factor * G + r

            if (s, a) in visited:
                continue
            
            visited.add((s, a))
            visit_counts[s, a] += 1

            # Update state-action value
            action_values[s, a] += G


    # Average action values
    action_values /= visit_counts + 1e-9

    return action_values

def every_visit_prediction(
    env: BaseMDP,
    policy: Agent,
    max_steps: int,
    num_episodes: int
) -> np.ndarray: 
    """Computes action-values via every-visit monte carlo prediction."""
    n_states = env.n_states
    n_actions = env.n_actions

    # Init action-values
    action_values = np.zeros((n_states, n_actions))
    visit_counts = np.zeros_like(action_values)

    for episode in range(num_episodes):
        # Sample trajectory
        trajectory = env.sample_trajectory(policy, max_steps)

        G = 0
        for (s, a, r, s_next) in trajectory[::-1]:
            # Update return
            G = env.discount_factor * G + r

            # Update state-action value
            action_values[s, a] += G

            # Increment visit counter
            visit_counts[s, a] += 1


    # Average action values
    action_values /= visit_counts + 1e-9

    return action_values
