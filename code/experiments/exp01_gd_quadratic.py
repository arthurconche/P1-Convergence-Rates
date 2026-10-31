import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from src.functions import A_drawing, quadratic, grad_quad, numerical_grad
from src.methods import grad_desc


SEED = 0
rng = np.random.default_rng(SEED)

"""
As A is defined positive, the associated quadratic function is supposed
to be minimized for x = 0
"""

# Parameters

kappas = [10, 100, 1000]
n_iter = 5000
d = 100

# Iterates computation

def run_gd_quadratic(kappa, d, n_iter, rng):
    mu =1; L = kappa
    A = A_drawing(d=d, mu=mu, L=L, rng=rng)
    f = lambda x : quadratic(x, A)
    grad_f = lambda x : grad_quad(x, A=A)
    x0 = rng.standard_normal(d)
    iterates = grad_desc(grad_f=grad_f, x0=x0, step = 1/L, n_iter = n_iter)

    fs = f(iterates)
    gs = grad_f(iterates)

    return fs, gs, L


# fig1a 01a - rks computation + plot

fig1a, ax1a = plt.subplots(1, len(kappas), figsize=(12,3), sharey=True)

fig1b, ax1b = plt.subplots(1, len(kappas), figsize=(12,3), sharey=True)
ks = np.arange(n_iter)

for i,kappa in enumerate(kappas):

    fs, gs, L = run_gd_quadratic(kappa = kappa, d = d, n_iter = n_iter, rng = rng)
    mask = fs[:-1] > 1e-12
    num = fs[:-1] - fs[1:]
    den = np.sum(gs[:-1]**2, axis = -1) / (2*L)
    rks = num[mask] / den[mask]


    # fig 1a

    ax1a[i].axhline(2 - 1/kappa, color = "black", linestyle =":", linewidth=1, label=r"upper bound $2 - 1/\kappa$ (Prop. 1)")
    ax1a[i].axhline(1, color="black", linestyle ="--", linewidth=1, label="lower bound 1 (Cor. 1)")

    ax1a[i].set_xscale("log")
    ax1a[i].plot(ks[mask]+1, rks, label = "measured")

    ax1a[i].set_title(f"$\\kappa = {kappa}$")

    # fig 1b


    th_bound = (1 - 1/kappa)**ks[mask] * ( fs[0] )
    ax1b[i].plot(ks[mask], th_bound, color = "black", linestyle ="--", linewidth=1, label=r"upper bound $(1-1/\kappa)^k\,(f(x_0)-f^\star)$")
    ax1b[i].set_yscale("log")

    ax1b[i].plot(ks[mask], fs[:-1][mask], label = "measured")

    ax1b[i].set_title(f"$\\kappa = {kappa}$")

# Labels fig1a

fig1a.supxlabel("iteration $k+1$")
fig1a.supylabel(r"$r_k$")

handles, labels = ax1a[0].get_legend_handles_labels()
fig1a.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5,0.0), ncol=3, frameon=False)
fig1a.tight_layout()

# Labels fig1b

fig1b.supxlabel("iteration $k$")
fig1b.supylabel(r"$f(x_k) - f^\star$")

handles, labels = ax1b[0].get_legend_handles_labels()
fig1b.legend(handles, labels, loc="upper center",bbox_to_anchor=(0.5,0.0), ncol=2, frameon=False)
fig1b.tight_layout()

# Saving figure

FIG_DIR = CODE_DIR /"figures"
FIG_DIR.mkdir(exist_ok=True)


fig1a.savefig(FIG_DIR/ "fig01a_rk.pdf", bbox_inches="tight")
plt.close(fig1a)

fig1b.savefig(FIG_DIR/ "fig01b_gap.pdf", bbox_inches="tight")
plt.close(fig1b)





