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

def nest_desc(grad_f, x0, step, n_iter, momentum):
    """
    Return the sequence of Nesterov descent iterates in array with shape (n_iter +1, d)
    """
    d = x0.size
    xs = np.zeros((n_iter + 1, d))
    xs[0,:] = x0
    x_prev=x0

    for i in range(n_iter):
        y = xs[i] + momentum *(xs[i] - x_prev)
        xs[i+1,:] = (y - step*grad_f(y))
        x_prev = xs[i]

    return xs
