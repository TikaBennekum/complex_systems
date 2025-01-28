"""
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file keeps track of important information from the simulations.
"""

import numpy as np
from numpy.typing import NDArray
from CA import BOTTOM_NEIGHBORS, CA


class CAHistory(CA):
    def __init__(self, width: int, height: int, initial_state: NDArray | None = None, neighbor_list=...):
        super().__init__(width, height, initial_state, neighbor_list)
        self.history = [self.grid]
    
    def update_grid(self):
        output = super().update_grid()
        self.history.append(np.copy(self.grid))
        return output
    
    def get_history(self):
        return np.array(self.history)
    
    def save_history(self, filename: str):
        np.save(filename, np.array(self.history))