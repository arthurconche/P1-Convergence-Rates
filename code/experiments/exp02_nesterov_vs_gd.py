import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from src.functions import A_drawing, grad_quad
from src.methods import grad_desc, nest_desc

SEED = 0
rng = np.random.default_rng(seed=SEED)

# Parameters
kappas = np.logspace(start=1, stop=4, num=30)
d = 100
mu = 1
n_drawing = 5
eps=1e-8



def nesterov_vs_gd(kappas, d, mu, rng, n_drawing, eps):
    """
    Computes iterates x_ks for gradient descent and nesterov descent
    Returns :
    ns_gd : average step amounts needed to hit eps for kappas with gradient descent.
    ns_nd : average step amounts needed to hit eps for kappas with Nesterov descent.
    """

    ns_gd, ns_nd = np.zeros(kappas.size), np.zeros(kappas.size)

    for i, kappa in enumerate(kappas):
        temp_ns_gd = np.zeros(n_drawing)
        temp_ns_nd = np.zeros(n_drawing)

        for j in range(n_drawing):
            L = kappa

            A = A_drawing(d=d, mu=mu, L=L, rng=rng)
            x0 = rng.standard_normal(d)
            grad_f = lambda x: grad_quad(X=x, A=A)

            n_iter_gd = int(20*kappa+1)
            traj_gd = grad_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter_gd)
            dist_gd = np.linalg.norm(traj_gd, axis=1)      # x* = 0
            hit_gd = dist_gd <= eps * dist_gd[0]
            assert hit_gd.any()
            n_gd = np.argmax(hit_gd)
            temp_ns_gd[j] = n_gd


            n_iter_nd = int(30*np.sqrt(kappa)+1)
            momentum = (np.sqrt(kappa) -1)/(np.sqrt(kappa)+1)
            traj_nd = nest_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter_nd, momentum=momentum)
            dist_nd = np.linalg.norm(traj_nd, axis=1)
            hit_nd = dist_nd <= eps * dist_nd[0]
            assert hit_nd.any()
            n_nd = np.argmax(hit_nd)
            temp_ns_nd[j] = n_nd


        ns_gd[i]=np.mean(temp_ns_gd)
        ns_nd[i]=np.mean(temp_ns_nd)

    return ns_gd, ns_nd

def fit_slope(x,y):
    """Slope of log(y) against log(x), fitted by least squares."""
    return np.polyfit(np.log(x), np.log(y), 1)[0]


if __name__ == "__main__":

    ns_gd, ns_nd = nesterov_vs_gd(kappas=kappas, d=d, mu=mu, rng=rng,n_drawing=n_drawing, eps=eps)
    slopes = {"gd":fit_slope(kappas, ns_gd), "nd":fit_slope(kappas, ns_nd)}

    fig, ax = plt.subplots(1,1)
    ax.loglog(kappas, ns_gd, label=f"gradient descent (slope {slopes['gd']:.2f})")
    ax.loglog(kappas, ns_nd, label=f"Nesterov (slope {slopes['nd']:.2f})")

    c_gd = np.median(ns_gd / kappas)
    ax.loglog(kappas, c_gd * kappas, "--", color="gray", label = r"$\propto \kappa$")
    c_nd = np.median(ns_nd / np.sqrt(kappas))
    ax.loglog(kappas, c_nd * np.sqrt(kappas), ":",  color="gray", label=r"$\propto \sqrt{\kappa}$")

    fig.supxlabel(r"condition number $\kappa$")
    fig.supylabel(rf"iterations to reach relative accuracy $\epsilon = 10^{-8}$")

    ax.legend()

    FIG_DIR = CODE_DIR /"figures"
    FIG_DIR.mkdir(exist_ok=True)


    fig.savefig(FIG_DIR/ "fig02_nesterov_vs_gd.pdf", bbox_inches="tight")
    plt.close(fig)