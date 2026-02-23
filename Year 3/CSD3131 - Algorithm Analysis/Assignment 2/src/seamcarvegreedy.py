# seamcarve.py — minimal DP seam-carving utilities
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".packages"))
import numpy as np
import cv2


def greedy_vertical_seam(cost: np.ndarray):
    """
    Myopic vertical seam: at each row, move to the lowest-cost neighbor
    among {j-1, j, j+1} (no dynamic programming).
    """
    H, W = cost.shape
    seam = [int(np.argmin(cost[0]))] # Start at the top row with the minimum cost pixel
    
    # Iterate through rows, selecting the minimum cost neighbor
    for i in range(1, H):
        j = seam[-1]
        
        # Determine valid neighbor range
        j_left = max(0, j - 1)
        j_right = min(W - 1, j + 1)
        
        # Find minimum cost neighbor
        min_cost = cost[i, j_left]
        j_new = j_left
        
        # Neighbor selection
        for jj in range(j_left + 1, j_right + 1):
            if cost[i, jj] < min_cost:
                min_cost = cost[i, jj]
                j_new = jj
        
        seam.append(int(j_new))
    
    return seam
