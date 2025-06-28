from utils.mdp import DeterministicMDP
import numpy as np

from typing import Tuple


class GridWorld(DeterministicMDP):
    """
    Implements a simple Grid World environment with absorbing boundaries and uniform step penalties, as given in the Sutton & Barto textbook.

    States are indexed in row-major order.
    Actions:
        [0, 1]  - Right
        [0, -1] - Left
        [1, 0]  - Down
        [-1, 0] - Up
    """
    def __init__(self, size: Tuple[int, int], discount_factor: float):
        # State Space
        n_states = np.prod(size)
        states = np.arange(n_states).reshape(size)

        # Action Space
        actions = np.array([
            [0, 1], [0, -1], [1, 0], [-1, 0]
        ]) 
        n_actions = 4
        
        # Deterministic Transition Function
        rows, cols = states.shape
        terminal_states_flat = [0, n_states-1]
        transitions = np.zeros((n_states, n_actions), dtype=np.int32)
        for s in range(n_states):
            if s in terminal_states_flat:    # absorbing state
                transitions[s, :] = s
                continue
            for a in range(n_actions):
                s_next = np.unravel_index(s, states.shape) + actions[a]

                if not (0 <= s_next[0] < rows and 0 <= s_next[1] < cols):   # off the grid
                    s_next = s
                else:
                    s_next = np.ravel_multi_index(s_next, states.shape)     # valid next state
                transitions[s, a] = s_next

        # Deterministic Rewards
        rewards = np.ones_like(transitions, dtype=np.float32) * -1.0

        # Initial State Distribution
        init_states = np.ones_like(states) / states.size

        super().__init__(
            states=states,
            actions=actions,
            transitions=transitions,
            rewards=rewards,
            init_states=init_states,
            discount_factor=discount_factor
        )

    def get_2d_coordinates(self, idx_flat: int) -> Tuple[int, int]:
        """Returns (row, col) coordinates for a flattened index in the grid."""
        return np.unravel_index(idx_flat, self.states.shape)

    def get_1d_index(self, idx_2d: Tuple[int, int]) -> int:
        """Returns flattened index for a 2D coordinate in the grid."""
        return np.ravel_multi_index(idx_2d, self.states.shape)
