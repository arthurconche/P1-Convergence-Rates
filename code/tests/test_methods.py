import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
from src.functions import A_drawing, quadratic, grad_quad
from src.methods import grad_desc, nest_desc, three_sequences





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
    traj_nd = nest_desc(grad_f=grad_f, x0=x0, L=L, n_iter=n_iter, momentum=0)

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
    traj_nd = nest_desc(grad_f=grad_f, x0=x0, L=L, n_iter=n_iter, momentum=momentum)

    f_gd = quadratic(traj_gd[-1], A=A)
    f_nes = quadratic(traj_nd[-1], A=A)

    assert  f_nes < 1e-6 * f_gd, f"Nesterov {f_nes:.2e}, GD : {f_gd:.2e}"

def test_three_sequence_equals_momentum():
    seed = 0
    rng = np.random.default_rng(seed)

    d = 10
    mu = 1
    kappas = [1, 10, 100]
    n_iter = 200

    for kappa in kappas:
        L = kappa*mu
        A = A_drawing(d=d, mu=mu, L=L, rng=rng)
        x0 = rng.standard_normal(d)
        beta = (np.sqrt(kappa) -1)/(np.sqrt(kappa)+1)
        grad_f = lambda x : grad_quad(x,A)

        traj_nd = nest_desc(grad_f=grad_f, x0=x0, L=L, n_iter=n_iter, momentum=beta)
        traj_three_seq , _ = three_sequences(grad_f=grad_f, x0=x0, L=L, mu=mu, n_iter=n_iter)

        assert np.allclose(traj_nd, traj_three_seq), f"nesterov descent and three sequences computes differents iterates for kappa={kappa}"

def test_potential_contraction():
    seed = 0
    rng = np.random.default_rng(seed)

    d = 10
    mu = 1
    kappas = [1, 10, 100]
    n_iter = 100



    for kappa in kappas:
        L = kappa*mu
        delta = 1/np.sqrt(kappa)
        A = A_drawing(d=d, mu=mu, L=L, rng=rng)
        x0 = rng.standard_normal(d)
        grad_f = lambda x : grad_quad(X=x, A=A)

        xs, zs = three_sequences(grad_f=grad_f, x0=x0, L=L, mu=mu, n_iter=n_iter)

        phis = quadratic(X=xs, A=A) + mu/2 * np.linalg.norm(zs, axis=1)**2
        ok = np.all(phis[1:] <= (1 - delta) * phis[:-1] + phis[0]*1e-10)

        assert ok, f"not okay for kappa={kappa}"
        





if __name__ == "__main__":
    test_nesterov_zero_momentum_is_gd()
    test_nesterov_accelerates()
    test_three_sequence_equals_momentum()
    test_potential_contraction()
    print('all tests passed')

