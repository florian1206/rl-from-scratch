"""Implement dynamic programming methods."""

import numpy as np

from utils.mdp import BaseMDP
from utils.agent import Agent

from typing import TypeAlias, Tuple

Values: TypeAlias = np.ndarray
Policy: TypeAlias = np.ndarray


def policy_evaluation(
        environment: BaseMDP,
        agent: Agent,
        epsilon: float,
        init_values: Values=None
) -> Values:
    """Given a model of the environment and a policy, the value of this policy is predicted iteratively."""

    # Init values
    values = init_values if init_values else np.zeros(environment.n_states)

    while True:
        expected_values = environment.transitions @ values
        expected_rewards = np.sum(environment.transitions * environment.rewards, axis=-1)
        action_values = expected_rewards + environment.discount_factor * expected_values

        new_values = np.einsum('sa,sa->s', agent.policy, action_values)
        
        # Stop Condition
        if np.max(np.abs(new_values - values)) < epsilon:
            return new_values
        
        values = new_values 


def policy_improvement(
        environment: BaseMDP,
        values: Values
) -> Policy:
    """Computes a new policy by greedily picking the best action for each state based on the given values."""
    # Compute action values
    expected_values = environment.transitions @ values
    expected_rewards = np.sum(environment.transitions * environment.rewards, axis=-1)
    action_values = expected_rewards + environment.discount_factor * expected_values

    # Compute greedy actions
    greedy_actions = np.argmax(action_values, axis=1) # (s,)

    # Create new policy
    policy = np.zeros((environment.n_states, environment.n_actions))
    policy[np.arange(policy.shape[0]), greedy_actions] = 1.0

    return policy


def policy_iteration(
        environment: BaseMDP,
        agent: Agent,
        epsilon: float,
) -> Tuple[Agent, Values]:
    """Computes the optimal value function within a given environment via the Policy Iteration algorithm."""

    while True:
        # Policy Evaluation
        values = policy_evaluation(environment, agent, epsilon)

        # Policy Improvement
        old_policy = agent.policy.copy()
        new_policy = policy_improvement(environment, values)
        agent.update(new_policy)

        # Stop Condition
        if np.array_equal(new_policy, old_policy):
            return agent, values
        

def value_iteration(
        environment: BaseMDP,
        epsilon: float,
        init_values: Values=None
) -> Tuple[Policy, Values]:
    """Computes the optimal value function within a given environment via the Value Iteration algorithm."""

    # Init values
    values = init_values if init_values else np.zeros(environment.n_states)

    while True:
        # Bellman Optimality Update
        expected_values = environment.transitions @ values
        expected_rewards = np.sum(environment.transitions * environment.rewards, axis=-1)
        action_values = expected_rewards + environment.discount_factor * expected_values

        new_values = np.max(action_values, axis=-1)
        
        # Stop Condition
        if np.max(np.abs(new_values - values)) < epsilon:
            break
        
        values = new_values 

    # Compute optimal policy
    policy = policy_improvement(environment, new_values)

    return policy, new_values
