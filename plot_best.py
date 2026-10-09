"""Plot measured vs free-run predicted output on the test segment, for one saved run."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import DT, N_TRAIN
name = sys.argv[1]
d = np.load(f"runs/{name}.npz")
t = np.arange(len(d["y"])) * DT
s = slice(N_TRAIN, None)
rmse = np.sqrt(np.mean((d["y_hat"][s] - d["y"][s]) ** 2))
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
ax[0].plot(t[s], d["y"][s], label="measured", lw=1.5)
ax[0].plot(t[s], d["y_hat"][s], "--", label="free-run prediction", lw=1.2)
ax[0].set_ylabel("y (V)"); ax[0].legend(); ax[0].set_title(f"{name}: test free-run RMSE = {rmse:.5f} V")
ax[1].plot(t[s], d["y_hat"][s] - d["y"][s], color="C3"); ax[1].set_ylabel("error (V)"); ax[1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig(f"test_plot_{name}.png", dpi=150)
