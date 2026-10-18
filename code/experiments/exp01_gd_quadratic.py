import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from src.functions import A_drawing, quadratic, grad_quad, numerical_grad
from src.methods import grad_desc


seed = 0
rng = np.random.default_rng(seed)

"""
As A is defined positive, the associated quadratic function is supposed
to be minimized for x = 0
"""

# Parametrisation
n=100
mu, L = 1, 100
n_iter = 500

# Iterates computation
A = A_drawing(n=n, mu=mu, L=L, rng=rng)
f = lambda x : quadratic(x, A)
grad_f = lambda x : grad_quad(x, A=A)
x0 = rng.standard_normal(n)
iterates = grad_desc(grad_f=grad_f, x0=x0, step = 1/L, n_iter = 500)

