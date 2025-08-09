"""Implements temporal difference learning methods."""

import numpy as np

from utils.agent import Agent
from utils.mdp import BaseMDP


def prediction(
    environment: BaseMDP,
    policy: Agent,
    alpha: float,
    num_episodes: int,
    max_steps: int
) -> np.ndarray:
    """Computes state-values using temporal-difference policy evaluation."""
    # Ensure valid learning rate
    assert 0 < alpha <= 1, f"Alpha must be in the range (0, 1], got {alpha}"

    n_states = environment.n_states
    gamma = environment.discount_factor
    terminal_states = set(environment.terminal_states)

    # Init state values
    state_values = np.zeros(n_states)

    for _ in range(num_episodes):
        state = environment.reset()
        for _ in range(max_steps):
            # Sample next state and reward
            action = policy.sample(state)
            next_state, reward = environment.step(state, action)

            # TD Update
            td_target = reward + gamma * state_values[next_state]
            state_values[state] += alpha * (td_target - state_values[state])

            # Stop Condition
            if next_state in terminal_states:
                break
            else:
                state = next_state
    
    return state_values
