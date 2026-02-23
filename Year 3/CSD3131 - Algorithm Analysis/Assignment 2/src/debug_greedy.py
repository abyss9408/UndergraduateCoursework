#!/usr/bin/env python3
"""Debug script: visualize energy map to understand what's happening."""
import numpy as np
import cv2
import sys

def sobel_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)

def greedy_vertical_seam(cost):
    H, W = cost.shape
    seam = [int(np.argmin(cost[0]))]
    for i in range(1, H):
        j = seam[-1]
        cand = [j]
        if j > 0:     cand.append(j-1)
        if j < W-1:   cand.append(j+1)
        j_new = min(cand, key=lambda jj: cost[i, jj])
        seam.append(int(j_new))
    return seam

# Load your image
img_path = "data/Broadway_tower_edit.jpg"
if len(sys.argv) > 1:
    img_path = sys.argv[1]

img = cv2.imread(img_path)
if img is None:
    print(f"Could not load {img_path}")
    sys.exit(1)

print(f"Loaded image: {img.shape[1]}×{img.shape[0]}")

# Compute energy
energy = sobel_energy(img)

# Normalize energy to 0-255 for visualization
energy_vis = ((energy - energy.min()) / (energy.max() - energy.min()) * 255).astype(np.uint8)
energy_vis_color = cv2.applyColorMap(energy_vis, cv2.COLORMAP_JET)

# Find greedy seam
seam = greedy_vertical_seam(energy)

# Visualize seam on original
img_with_seam = img.copy()
for i, j in enumerate(seam):
    img_with_seam[i, j] = (0, 255, 0)  # Green seam
    # Make it thicker for visibility
    if j > 0:
        img_with_seam[i, j-1] = (0, 255, 0)
    if j < img.shape[1]-1:
        img_with_seam[i, j+1] = (0, 255, 0)

# Visualize seam on energy map
energy_with_seam = energy_vis_color.copy()
for i, j in enumerate(seam):
    energy_with_seam[i, j] = (0, 255, 0)
    if j > 0:
        energy_with_seam[i, j-1] = (0, 255, 0)
    if j < img.shape[1]-1:
        energy_with_seam[i, j+1] = (0, 255, 0)

# Save outputs
cv2.imwrite('debug_energy_map.png', energy_vis_color)
cv2.imwrite('debug_seam_on_image.png', img_with_seam)
cv2.imwrite('debug_seam_on_energy.png', energy_with_seam)

print(f"\nSaved visualizations:")
print(f"  Energy map (hot=high energy):     debug_energy_map.png")
print(f"  Seam on original image (green):   debug_seam_on_image.png")
print(f"  Seam on energy map (green):       debug_seam_on_energy.png")
print(f"\nEnergy statistics:")
print(f"  Min:  {energy.min():.1f}")
print(f"  Max:  {energy.max():.1f}")
print(f"  Mean: {energy.mean():.1f}")
print(f"\nSeam total energy: {sum(energy[i, seam[i]] for i in range(len(seam))):.1f}")
print(f"\nLook at the energy map (hot colors = high energy).")
print(f"The green seam should go through COOL colors (blue/dark = low energy).")
