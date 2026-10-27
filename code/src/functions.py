import numpy as np

def A_drawing(d, mu:float, L:float, rng):
    """
    Returns a "random" symmetric, definite positive, d*d matrix A with eigenvalues in [mu, L]
    """
    if d<2:
        raise ValueError("n needs to be >= 2")
    
    eigenvalues = mu+(L-mu)*rng.random(d-2)       # eigenvalues tirage
    eigenvalues = np.append(eigenvalues, [mu,L])
    D = np.diag(eigenvalues)

    gauss_mat = rng.standard_normal((d, d))
    Q,R = np.linalg.qr(gauss_mat)
    A = Q @ D @ Q.T                                 # A construction

    return A

def quadratic(X, A):
    """
    Computes the quadratic operator 1/2 * X.T @ A @ X.
    X : np.array (d,) or (N,d)
    Output : float64 or array (N,)
    """
    return (1/2 * X * (X @ A)).sum(axis=-1)

def grad_quad(X, A):
    """
    Return gradient for quadratic 1/2 * X.T @ A @ X function.
    A is supposed symmetric.
    X : np.array (d,) or (N,d)
    Output : np.array (d,) or (N,d)

    """
    return X @ A

def numerical_grad(f,x, eps=1e-6):
    d=x.size
    df=np.zeros(d)
    ens = np.eye(d)
    for i in range(d):
        en=ens[i,:]
        df[i]=(f(x+eps*en)-f(x-eps*en))/(2*eps)

    return df


