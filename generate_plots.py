import numpy as np
import tadasets
import matplotlib.pyplot as plt
from ripser import ripser
import persim
import os

# Set seed and generate data
np.random.seed(4242)
data_clean = tadasets.dsphere(d=1, n=1000, noise=0.0)
data_noisy = tadasets.dsphere(d=1, n=1000, noise=0.2)

# Create plots directory if it doesn't exist
output_dir = "/Users/aaronhaddad/Downloads/Statapp-1"
os.makedirs(output_dir, exist_ok=True)

# 1. Plot the spheres
plt.figure(figsize=(8, 8))
plt.scatter(data_clean[:,0], data_clean[:,1], label="Clean data (no noise)", alpha=0.7)
plt.scatter(data_noisy[:,0], data_noisy[:,1], label="Noisy data (noise=0.2)", alpha=0.5)
plt.axis('equal')
plt.legend(fontsize=12)
plt.title("Synthetic 1D Spheres embedded in 2D", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "synthetic_spheres.png"), dpi=300)
plt.close()

# 2. Compute and plot persistence diagrams
dgms_clean = ripser(data_clean)['dgms']
dgms_noisy = ripser(data_noisy)['dgms']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
persim.plot_diagrams(dgms_clean, show=False)
plt.title("Persistence Diagram: Clean Sphere", fontsize=14)

plt.subplot(1, 2, 2)
persim.plot_diagrams(dgms_noisy, show=False)
plt.title("Persistence Diagram: Noisy Sphere", fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "persistence_diagrams_spheres.png"), dpi=300)
plt.close()

print("Plots successfully saved to:", output_dir)
