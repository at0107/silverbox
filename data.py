"""Step 1: load the Silverbox excerpt, center it, split it into train / test."""
import numpy as np
import scipy.io as sio

FS = 610.35            # sampling frequency in Hz (from the homework)
DT = 1.0 / FS          # time between two samples, in seconds
START, STOP = 42650, 46746   # the 4096-sample slice
N_TRAIN, N_TEST = 3072, 1024


def load_excerpt(path="SNLS80mV.mat"):
    m = sio.loadmat(path)
    u = m["V1"].ravel()[START:STOP].astype(np.float64)   # input voltage
    y = m["V2"].ravel()[START:STOP].astype(np.float64)   # output voltage
    # center: subtract the mean of THIS excerpt (as the homework says)
    u = u - u.mean()
    y = y - y.mean()
    return u, y


def finite_diff_velocity(y):
    """v_k ~ (y_{k+1} - y_{k-1}) / (2 dt): central difference, one-sided at the ends."""
    return np.gradient(y, DT)


if __name__ == "__main__":
    u, y = load_excerpt()
    assert len(u) == N_TRAIN + N_TEST
    v = finite_diff_velocity(y)
    te = slice(N_TRAIN, None)
    print(f"samples: {len(u)}  dt = {DT*1e3:.4f} ms")
    print(f"train duration {N_TRAIN*DT:.2f} s, test duration {N_TEST*DT:.2f} s")
    print(f"u: std {u.std():.4f}  max|u| {abs(u).max():.4f}")
    print(f"y: std {y.std():.4f}  max|y| {abs(y).max():.4f}")
    print(f"v: std {v.std():.2f}  max|v| {abs(v).max():.1f}")
    print(f"zero-prediction baseline (RMS of test y): {np.sqrt(np.mean(y[te]**2)):.4f}")
    # dominant frequency of y, to check the "near 70 Hz" claim
    f = np.fft.rfftfreq(len(y), DT)
    print(f"dominant frequency of y: {f[np.argmax(abs(np.fft.rfft(y)))]:.1f} Hz")
