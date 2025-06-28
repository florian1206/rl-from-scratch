from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from typing import Tuple


@dataclass
class BaseMDP(ABC):
    """Blueprint for a Markov Decision Process."""
    states: np.ndarray
    actions: np.ndarray
    init_states: np.ndarray
    discount_factor: float = 1

    n_states: int = field(init=False)
    n_actions: int = field(init=False)    

    def __post_init__(self):
        self.n_states = np.prod(self.states.shape)
        self.n_actions = np.prod(self.actions.shape)

        # Ensure the initial states define a proper probability distribution
        assert np.all(self.init_states >= 0), "Initial state distribution must be non-negative."
        assert np.isclose(self.init_states.sum(), 1.0), "Initial state distribution must sum to 1."

    @abstractmethod
    def step(self, state: int, action: int) -> Tuple[int, float]:
        """Determines the next state, based on the current state and action."""
        pass

    def reset(self) -> int:
        """Chooses initial state from initial state distribution"""
        return np.random.choice(self.n_states, p=self.init_states)


@dataclass
class DeterministicMDP(BaseMDP):
    """Implements a deterministic Markov Decision Process."""
    transitions: np.ndarray        # (n_states, n_actions)
    rewards: np.ndarray            # (n_states, n_actions)

    def step(self, state: int, action: int) -> Tuple[int, float]:
        """Deterministically determines the next state, based on the current state and action."""
        s_next = self.transitions[state, action]
        r = float(self.rewards[state, action])

        return s_next, r
        

@dataclass
class StochasticMDP(BaseMDP):
    """Implements a stochastic Markov Decision Process."""
    transitions: np.ndarray        # (n_states, n_actions, n_states)
    rewards: np.ndarray            # (n_states, n_actions, n_states)

    def __post_init__(self):
        super().__post_init__()

        # Ensure the transition function is a proper probability distribution
        assert np.all(self.transitions >= 0), "Transition probabilities must be non-negative."
        assert np.allclose(np.sum(self.transitions, axis=-1), 1.0), "Transition probabilities must sum to 1."

    def step(self, state: int, action: int) -> Tuple[int, float]:
        """Stochastically determines the next state, based on the current state and action."""        
        s_next = np.random.choice(self.n_states, p=self.transitions[state, action])
        r = float(self.rewards[state, action, s_next])

        return s_next, r
