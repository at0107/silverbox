"""Final plot: measured vs free-run predicted y on the test segment only (chosen model a12, seed 0)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import DT, N_TRAIN

d = np.load("runs/s2_a12_seed0.npz")
s = slice(N_TRAIN, None)
t = np.arange(len(d["y"]))[s] * DT          # seconds from the start of the excerpt
rmse = np.sqrt(np.mean((d["y_hat"][s] - d["y"][s]) ** 2))
plt.figure(figsize=(9, 4))
plt.plot(t, d["y"][s], label="measured y", lw=1.5)
plt.plot(t, d["y_hat"][s], "--", label="free-run prediction", lw=1.2)
plt.xlabel("time (s)"); plt.ylabel("y (V, centered)")
plt.title(f"Silverbox test segment, free run: test RMSE = {rmse:.5f} V")
plt.legend(); plt.tight_layout(); plt.savefig("final_test_plot.png", dpi=150)
print(f"test RMSE {rmse:.5f}")
