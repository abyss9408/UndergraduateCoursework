# seamcarve.py — minimal DP seam-carving utilities
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".packages"))
import numpy as np
import cv2


def min_vertical_seamdp(cost: np.ndarray, img: np.ndarray):
    """DP lowest-cost vertical seam using forward energy.

    Args:
        cost: Energy map (H x W) - not used in pure forward energy but kept for compatibility
        img: Original image (H x W x 3) - required for forward energy computation

    Returns:
        List of column indices representing the seam path from top to bottom
    """
    H, W = cost.shape
    # Space optimization: only keep 2 rows of M (previous and current)
    M = np.zeros((2, W), dtype=np.float32)
    P = np.zeros((H, W), dtype=np.int8)

    # Convert image to grayscale for forward energy computation
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    # Initialize first row (row 0)
    M[0, :] = 0

    # Dynamic programming - forward energy only
    for i in range(1, H):
        prev = (i - 1) % 2  # Previous row index (0 or 1)
        curr = i % 2        # Current row index (0 or 1)

        for j in range(W):
            # Base cost of joining left and right neighbors
            if j > 0 and j < W-1:
                c_base = abs(gray[i, j+1] - gray[i, j-1])
            else:
                c_base = 0

            if j == 0:
                # Left edge
                c_u = abs(gray[i, j+1] - gray[i, j]) if j < W-1 else 0
                c_r = c_base + abs(gray[i-1, j] - gray[i, j+1]) if j < W-1 else float('inf')

                costs = [
                    M[prev, j] + c_u,
                    M[prev, j+1] + c_r
                ]
                idx = np.argmin(costs)
                M[curr, j] = costs[idx]
                P[i, j] = idx  # 0=straight, 1=right

            elif j == W-1:
                # Right edge
                c_l = c_base + abs(gray[i-1, j] - gray[i, j-1]) if j > 0 else float('inf')
                c_u = abs(gray[i, j-1] - gray[i, j]) if j > 0 else 0

                costs = [
                    M[prev, j-1] + c_l,
                    M[prev, j] + c_u
                ]
                idx = np.argmin(costs)
                M[curr, j] = costs[idx]
                P[i, j] = -1 if idx == 0 else 0  # -1=left, 0=straight

            else:
                # Interior: forward energy
                c_l = c_base + abs(gray[i-1, j] - gray[i, j-1])
                c_u = c_base
                c_r = c_base + abs(gray[i-1, j] - gray[i, j+1])

                w0 = M[prev, j-1] + c_l
                w1 = M[prev, j] + c_u
                w2 = M[prev, j+1] + c_r

                # Prefer center when equal to avoid bias
                if w1 <= w0 and w1 <= w2:
                    k, wmin = 0, w1
                elif w0 <= w2:
                    k, wmin = -1, w0
                else:
                    k, wmin = +1, w2

                P[i, j] = k
                M[curr, j] = wmin

    # Find minimum cost column in bottom row (last computed row)
    last_row = (H - 1) % 2
    j = int(np.argmin(M[last_row]))

    # Backtrack to find seam path
    seam = [j]
    for i in range(H-1, 0, -1):
        j = j + int(P[i, j])
        seam.append(j)
    seam.reverse()

    return seam

def remove_vertical_seamdp(img: np.ndarray, seam):
    """Remove the provided vertical seam from a color image."""
    H, W = img.shape[:2]
    out = np.empty((H, W-1, 3), dtype=img.dtype)
    for i, j in enumerate(seam):
        out[i] = np.delete(img[i], j, axis=0)
    return out
