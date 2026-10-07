"""Neural-ODE for the Silverbox:  y' = v,  v' = alpha_theta(y, v, u),  integrated with RK4.

Usage:  python silverbox_pinn.py '{"name": "tanh64x3", "width": 64, "depth": 3, ...}'
Each run appends one line to results.jsonl and saves its free-run prediction to runs/<name>.npz
"""
import json, sys, time, math, os
import numpy as np
import torch
import torch.nn as nn

from data import load_excerpt, finite_diff_velocity, DT, N_TRAIN, N_TEST

torch.set_num_threads(1)   # many small runs in parallel beat one run using all cores


# ----------------------------------------------------------------------------
# 1. The network: (y, v, u) -> acceleration
# ----------------------------------------------------------------------------
class AccelNet(nn.Module):
    """MLP on *scaled* inputs. Raw numbers differ by orders of magnitude
    (y ~ 0.05 V, v ~ 20 V/s, acceleration ~ 1e4 V/s^2), so we divide each input by
    a typical size and multiply the output by a typical acceleration size.
    That makes every number the optimizer sees close to 1."""

    def __init__(self, scales, width=64, depth=3, act="tanh", skip=False, phys_init=False):
        super().__init__()
        self.ys, self.vs, self.us, self.acc_s = scales
        acts = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        layers, n_in = [], 3
        for _ in range(depth):
            layers += [nn.Linear(n_in, width), acts[act]()]
            n_in = width
        layers.append(nn.Linear(n_in, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].bias)
        self.mlp[-1].weight.data *= 0.1          # start with a small nonlinear part
        # optional linear "skip" path: acceleration = mlp(...) + W [y, v, u]
        self.skip = nn.Linear(3, 1, bias=False) if skip else None
        if skip:
            nn.init.zeros_(self.skip.weight)
            if phys_init:
                # start from a linear oscillator at 70 Hz: v' = -w0^2 y   (in scaled units)
                w0 = 2 * math.pi * 70.0
                self.skip.weight.data[0, 0] = -(w0 ** 2) * self.ys / self.acc_s

    def forward(self, y, v, u):
        x = torch.stack([y / self.ys, v / self.vs, u / self.us], dim=-1)
        out = self.mlp(x).squeeze(-1)
        if self.skip is not None:
            out = out + self.skip(x).squeeze(-1)
        return out * self.acc_s


# ----------------------------------------------------------------------------
# 2. RK4 step.  u is only measured at the samples, so for the half step we use the
#    average of u_k and u_{k+1} (linear interpolation).
# ----------------------------------------------------------------------------
def rk4_step(net, y, v, u0, u1, dt):
    um = 0.5 * (u0 + u1)
    def f(y_, v_, u_):
        return v_, net(y_, v_, u_)
    k1y, k1v = f(y, v, u0)
    k2y, k2v = f(y + 0.5 * dt * k1y, v + 0.5 * dt * k1v, um)
    k3y, k3v = f(y + 0.5 * dt * k2y, v + 0.5 * dt * k2v, um)
    k4y, k4v = f(y + dt * k3y, v + dt * k3v, u1)
    y_new = y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_new, v_new


def rollout(net, y0, v0, u_seq, dt):
    """y0, v0: (B,)   u_seq: (B, L+1)  ->  predicted y and v at steps 0..L, shape (B, L+1)."""
    ys, vs = [y0], [v0]
    y, v = y0, v0
    for k in range(u_seq.shape[1] - 1):
        y, v = rk4_step(net, y, v, u_seq[:, k], u_seq[:, k + 1], dt)
        ys.append(y); vs.append(v)
    return torch.stack(ys, 1), torch.stack(vs, 1)


# ----------------------------------------------------------------------------
# 3. The metric: one free run over the entire excerpt (never reads y after sample 0)
# ----------------------------------------------------------------------------
@torch.no_grad()
def free_run(net, u, y, v_fd):
    u_t = torch.tensor(u, dtype=torch.float32)[None]
    yh, _ = rollout(net, torch.tensor([y[0]], dtype=torch.float32),
                    torch.tensor([v_fd[0]], dtype=torch.float32), u_t, DT)
    yh = yh[0].numpy()
    if not np.all(np.isfinite(yh)):
        yh = np.nan_to_num(yh, nan=1e3, posinf=1e3, neginf=-1e3)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    return yh, rmse(yh[:N_TRAIN], y[:N_TRAIN]), rmse(yh[N_TRAIN:], y[N_TRAIN:])


# ----------------------------------------------------------------------------
# 4. Training with random windows ("multiple shooting")
# ----------------------------------------------------------------------------
def run(cfg):
    c = dict(name="run", width=64, depth=3, act="tanh", skip=False, phys_init=False,
             opt="adam", lr=3e-3, steps=3000, win=64, batch=64, seed=0,
             lr_final=1e-4, win_final=None, clip=1.0)
    c.update(cfg)
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])

    u, y = load_excerpt()
    v_fd = finite_diff_velocity(y)
    scales = (float(y[:N_TRAIN].std()), float(v_fd[:N_TRAIN].std()), float(u[:N_TRAIN].std()),
              float(v_fd[:N_TRAIN].std() * 2 * math.pi * 70))
    net = AccelNet(scales, c["width"], c["depth"], c["act"], c["skip"], c["phys_init"])
    params = list(net.parameters())
    n_params = sum(p.numel() for p in params)
    opt = {"adam": lambda: torch.optim.Adam(params, lr=c["lr"]),
           "adamw": lambda: torch.optim.AdamW(params, lr=c["lr"], weight_decay=1e-4),
           "rmsprop": lambda: torch.optim.RMSprop(params, lr=c["lr"])}[c["opt"]]()
    # learning rate decays smoothly (cosine) from lr to lr_final
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c["steps"], eta_min=c["lr_final"])

    # training data as tensors (ONLY the training record is used for fitting)
    ut = torch.tensor(u[:N_TRAIN], dtype=torch.float32)
    yt = torch.tensor(y[:N_TRAIN], dtype=torch.float32)
    vt = torch.tensor(v_fd[:N_TRAIN], dtype=torch.float32)

    t0 = time.time()
    for step in range(c["steps"]):
        # window length grows linearly from `win` to `win_final` (curriculum)
        L = c["win"] if c["win_final"] is None else int(
            c["win"] + (c["win_final"] - c["win"]) * step / max(1, c["steps"] - 1))
        starts = torch.randint(0, N_TRAIN - L, (c["batch"],))
        idx = starts[:, None] + torch.arange(L + 1)[None, :]           # (B, L+1)
        yh, _ = rollout(net, yt[starts], vt[starts], ut[idx], DT)
        loss = ((yh - yt[idx]) ** 2).mean() / scales[0] ** 2            # relative MSE
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, c["clip"])
        opt.step(); sched.step()
        if step % 500 == 0 or step == c["steps"] - 1:
            print(f"[{c['name']}] step {step:5d}  L={L:4d}  loss {loss.item():.3e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    yh, tr, te = free_run(net, u, y, v_fd)
    res = dict(c, n_params=n_params, train_rmse=tr, test_rmse=te, seconds=time.time() - t0)
    print(f"[{c['name']}] FREE RUN  train RMSE {tr:.5f}   test RMSE {te:.5f}", flush=True)
    os.makedirs("runs", exist_ok=True)
    np.savez(f"runs/{c['name']}.npz", y_hat=yh, y=y, u=u)
    torch.save({"state": net.state_dict(), "scales": scales, "cfg": c}, f"runs/{c['name']}.pt")
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


if __name__ == "__main__":
    run(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
