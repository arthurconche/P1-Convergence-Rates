import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.functions import A_drawing, quadratic, grad_quad
from src.methods import three_sequences

SEED = 0
rng = np.random.default_rng(SEED)

kappas = [10, 100, 1000]
d = 100
mu = 1

def run_nesterov_quadratic(kappa, mu, d, n_iter, rng) :
    L = kappa * mu
    A = A_drawing(d=d, mu=mu, L=L, rng=rng)
    grad_f = lambda x : grad_quad(X=x, A=A)
    x0 = rng.standard_normal(d)

    xs, zs = three_sequences(grad_f=grad_f, x0=x0, L=L, mu=mu, n_iter=n_iter)

    fs = quadratic(X=xs, A=A)
    phis = quadratic(X=xs, A=A) + mu/2 * np.linalg.norm(zs, axis =1)**2

    return fs, phis, L

def fit_slope(x,y):
    """Slope of log(y) against x, fitted by least squares."""
    return np.polyfit(x, np.log(y), 1)[0]

# Plot

if __name__ == "__main__":
    fig, ax = plt.subplots(1, len(kappas), figsize=(9,2.8), sharey=True)

    results = []
    for i, kappa in enumerate(kappas) :
        n_iter = int(30 * np.sqrt(kappa))
        ks = np.arange(n_iter+1)
        fs, phis, L = run_nesterov_quadratic(kappa=kappa, mu=mu, d=d, n_iter = n_iter, rng = rng)
        bounds=phis[0]*(1-1/np.sqrt(kappa))**ks
        slope_measured = fit_slope(ks, fs)
        slope_bound = fit_slope(ks,bounds)
        r = slope_measured / slope_bound

        results.append({"kappa": kappa,
                        "slope_measured":slope_measured,
                        "slope_bound":slope_bound,
                        "ratio":r})



        ax[i].plot(ks, fs, 'r', label=r"measured $f-f^*$")
        ax[i].plot(ks, phis, 'black', linewidth=1, label =r"measured $\Phi_k$")
        ax[i].plot(ks, bounds, ls = '--', label=r"upper bound $(1-1/\sqrt{\kappa})^k \Phi_0$")
        ax[i].set_yscale('log')
        ax[i].set_title(f"$\\kappa = {kappa}$")


    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center",bbox_to_anchor=(0.5,0.0), ncol=3, frameon=False)

    fig.supxlabel("iteration $k$")
    fig.supylabel(r"$f(x_k) - f^\star$, $\Phi_k$")
    plt.tight_layout()

    FIG_DIR = CODE_DIR / "figures"
    fig.savefig(FIG_DIR / "fig03_nesterov_bound.pdf", bbox_inches="tight")


    df = pd.DataFrame(results)
    print(df.round(4).to_string(index=False))

    plt.show()

