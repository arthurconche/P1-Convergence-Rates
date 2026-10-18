import numpy as np

def A_drawing(n, mu:float, L:float, rng):
    """
    Returns a "random" symmetric, definite positive, n*n matrix A with eigenvalues in [mu, L]
    """
    if n<2:
        raise ValueError("n needs to be >= 2")
    
    eigenvalues = mu+(L-mu)*rng.random(n-2)       # eigenvalues tirage
    eigenvalues = np.append(eigenvalues, [mu,L])
    D = np.diag(eigenvalues)

    gauss_mat = rng.standard_normal((n, n))
    Q,R = np.linalg.qr(gauss_mat)
    A = Q @ D @ Q.T                                 # A construction

    return A

def quadratic(X, A):
    """
    Computes the quadratic operator 1/2 * x.T @ A @ x.
    """
    return (1/2 * X * (X @ A)).sum(axis=-1)

def grad_quad(X, A):
    """
    Return gradient for quadratic 1/2 * x.T @ A @ x function.
    As A is symmetric, A.T = A
    """
    return X @ A

def numerical_grad(f,x, eps=1e-6):
    n=x.size
    df=np.zeros(n)
    ens = np.eye(n)
    for i in range(n):
        en=ens[i,:]
        df[i]=(f(x+eps*en)-f(x-eps*en))/(2*eps)

    return df


