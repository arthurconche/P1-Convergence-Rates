import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
from src.functions import A_drawing, quadratic, grad_quad
from src.methods import grad_desc, nest_desc





def test_nesterov_zero_momentum_is_gd():
    seed=0
    rng = np.random.default_rng(seed)
    d = 10
    mu = 1
    L = 100
    A = A_drawing(d=d, mu=mu, L=L, rng=rng)
    x0 = rng.standard_normal(d)

    n_iter = 1000
    grad_f = lambda x : grad_quad(x,A)


    traj_gd = grad_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter)
    traj_nd = nest_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter, momentum=0)

    assert np.allclose(traj_gd, traj_nd), "momentum=0 should give gradient descent"



def test_nesterov_accelerates():
    seed=0
    rng = np.random.default_rng(seed)

    d = 10
    kappa = 100
    mu = 1
    L = kappa
    A = A_drawing(d=d, mu=mu, L=L, rng=rng)
    x0 = rng.standard_normal(d)

    n_iter = 200
    momentum = (np.sqrt(kappa) -1)/(np.sqrt(kappa)+1)
    grad_f = lambda x : grad_quad(x,A)


    traj_gd = grad_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter)
    traj_nd = nest_desc(grad_f=grad_f, x0=x0, step=1/L, n_iter=n_iter, momentum=momentum)

    f_gd = quadratic(traj_gd[-1], A=A)
    f_nes = quadratic(traj_nd[-1], A=A)

    assert  f_nes < 1e-6 * f_gd, f"Nesterov {f_nes:.2e}, GD : {f_gd:.2e}"
    
    

if __name__ == "__main__":
    test_nesterov_zero_momentum_is_gd()
    test_nesterov_accelerates()
    print('all tests passed')