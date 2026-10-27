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


# rks computation + plot

fig, ax = plt.subplots(1, len(kappas), figsize=(12,3), sharey=True)
for i,kappa in enumerate(kappas):

    fs, gs, L = run_gd_quadratic(kappa = kappa, d = d, n_iter = n_iter, rng = rng)
    mask = fs[:-1] > 1e-12
    num = fs[:-1] - fs[1:]
    den = np.sum(gs[:-1]**2, axis = -1) / (2*L)
    rks = num[mask] / den[mask]
    ks = np.arange(n_iter)

    ax[i].axhline(2 - 1/kappa, color = "red", linestyle =":", linewidth=1, label="upper bound : $2 - \\frac{1}{\\kappa}$")
    ax[i].axhline(1, color="orange", linestyle ="--", linewidth=1, label="lower bound : 1")

    ax[i].set_xscale("log")
    ax[i].plot(ks[mask]+1, rks)
    ax[i].set_xlabel("iteration $k+1$")
    ax[i].set_title(f"$\\kappa = {kappa}$")


ax[0].set_ylabel("$r_k$")

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.08, 1, 1))

# Saving figure

FIG_DIR = CODE_DIR /"figures"
FIG_DIR.mkdir(exist_ok=True)

fig.savefig(FIG_DIR/ "fig01a_rk.pdf", bbox_inches="tight")
plt.close(fig)