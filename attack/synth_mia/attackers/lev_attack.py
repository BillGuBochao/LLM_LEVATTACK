import numpy as np
from ..base import BaseAttacker
from scipy.spatial import cKDTree
from typing import Dict, Any, Tuple, List, Optional
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

class levDCR(BaseAttacker):
    def __init__(self, hyper_parameters=None):
        if hyper_parameters is None:
            hyper_parameters = {}
        super().__init__(hyper_parameters)
        self.name = "levDCR"

    def row_to_str(row):
        return ','.join(
            str(round(float(val), 5)) if isinstance(val, (int, float, np.number)) else str(val)
            for val in row
        )

    def compute_min_distances(self, test_rows, synth_rows):
        # Flatten the arrays to get 1D strings
        test_strings = [row[0] if isinstance(row, np.ndarray) else row for row in test_rows]
        synth_strings = [row[0] if isinstance(row, np.ndarray) else row for row in synth_rows]
        
        return [
            -min(Levenshtein.distance(test_row, synth_row) for synth_row in synth_strings)
            for test_row in tqdm(test_strings, desc="Computing distances")
        ]

    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the attack scores using efficient nearest neighbor search.
        
        Args:
            X_test (np.ndarray): Test data (member and non-member)
            synth (np.ndarray): Synthetic data
            ref (np.ndarray): Reference data (not used in this implementation)
        
        Returns:
            np.ndarray: Predicted scores
        """

        
        nearest_distances = self.compute_min_distances(X_test, synth)
        scores =  (nearest_distances)
        return scores