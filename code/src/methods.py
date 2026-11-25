import numpy as np

def grad_desc(grad_f, x0, step, n_iter):
    """
    Return the sequence of gradient descent iterates in array with shape (n_iter + 1, d)
    """
    d = x0.size
    xs = np.zeros((n_iter + 1, d))
    xs[0,:] = x0
    
    
    for i in range(n_iter):
        x_new = xs[i,:] - step * grad_f(xs[i,:])
        xs[i+1,:]=(x_new)

    return xs

def nest_desc(grad_f, x0, L, n_iter, momentum):
    """
    Return the sequence of Nesterov descent iterates in array with shape (n_iter +1, d)
    """
    d = x0.size
    xs = np.zeros((n_iter + 1, d))
    xs[0,:] = x0
    x_prev=x0

    for k in range(n_iter):
        y = xs[k] + momentum *(xs[k] - x_prev)
        xs[k+1,:] = (y - 1/L*grad_f(y))
        x_prev = xs[k]

    return xs

def three_sequences(grad_f, x0, L, mu, n_iter):
    """Return the sequence of the three sequences algorithm iterates in arrays with shape (n_iter+1,d)"""

    # Constantes
    kappa = L/mu
    tau = 1/(1+np.sqrt(kappa))
    delta = 1/np.sqrt(kappa)

    # Compute iterates
    d = x0.size
    xs = np.zeros((n_iter + 1, d))
    zs = np.zeros((n_iter + 1, d))
    xs[0], zs[0] = x0, x0

    for k in range(n_iter):
        y = (1-tau)*xs[k] + tau*zs[k]
        g = grad_f(y)
        xs[k+1] = y - g*(1/L)
        zs[k+1] = (1-delta)*zs[k] + delta*(y - g*(1/mu))

    return xs, zs
