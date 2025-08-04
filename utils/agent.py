from dataclasses import dataclass
import numpy as np


@dataclass
class Agent:
    """Implements an agent that can interact with an environment. Ensures compatibility with environment for learning algorithms."""
    policy: np.ndarray
    n_actions: int

    @classmethod
    def create(cls, n_states: int, n_actions: int):
        """Alternative constructor to initialize the agent with a uniform random policy."""
        policy = np.ones((n_states, n_actions)) * (1 / n_actions)
        return cls(policy, n_actions)

    def __post_init__(self):
        assert self.policy.ndim == 2, "Policy must be of shape (n_states, n_actions)"
        assert self.policy.shape[1] == self.n_actions, "Policy does not match supplied number of actions"

    def sample(self, state: int) -> int:
        """Sample an action for a given state under the agent's current policy."""
        return np.random.choice(self.n_actions, p=self.policy[state])
    
    def update(self, policy_new: np.ndarray) -> None:
        """Update the agent's policy."""
        self.policy = policy_new

    def get_action_dist(self, state: int) -> np.ndarray:
        """Get the conditional probability distribution across actions for a given state."""
        return self.policy[state]