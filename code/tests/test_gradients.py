import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from src.functions import A_drawing, quadratic, grad_quad, numerical_grad

def test_gradient_check():
    seed = 0
    rng = np.random.default_rng(seed)

    # Compute numerical grad for quadratic function

    A = A_drawing(d=10, mu=1, L=100, rng=rng)
    f = lambda x : quadratic(x,A)

    xs=rng.standard_normal((10,10))
    err = []

    for x in xs:
        true_grad=grad_quad(x, A=A)
        num_grad=numerical_grad(f=f,x=x)
        err.append(np.linalg.norm(true_grad-num_grad)/np.linalg.norm(true_grad))

    max_err = max(err)

    assert max_err < 3e-5, "gradient is wrong"

if __name__ == "__main__":
    test_gradient_check()
    print("All tests passed")