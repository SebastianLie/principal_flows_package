from common_methods_sphere import log_map_sphere, exp_map_sphere
import numpy as np
import math

# ideas: plot schilds ladder using ax.quiver(xyz of points, xyz of vector)

def schilds_ladder_hypersphere(p, p_prime, vector):
    """[summary]

    Args:
        p ([type]): [description]
        p_prime ([type]): [description]
        vector ([type]): [description]

    Returns:
        [type]: [description]
    """    
    vector_normalised = vector/np.linalg.norm(vector)
    X0 = exp_map_sphere(p, vector_normalised)
    g = log_map_sphere(p, p_prime)
    dist_g = np.linalg.norm(g)
    step_up = 0.1
    N = math.ceil(dist_g/step_up) + 1
    step = dist_g/N
    e = step * g / dist_g
    A = np.zeros(shape=(3, N+1))
    A[:, 0] = p
    for i in range(N):
        A[:, i+1] = exp_map_sphere(A[:, i], e)
    #A[ ,N+1]=A1
    # if we don't copy, i.e X = A, then they refer to the same array 
    # or address in mem, and any change to X is a change to A.....
    X = np.copy(A)
    X[:, 0] = X0
    for j in range(N):
        t1 = log_map_sphere(A[:, j+1], X[:, j])
        P = exp_map_sphere(A[:, j+1], 0.5*t1)
        t2 = log_map_sphere(A[:, j], P)
        X[:, j+1] = exp_map_sphere(A[:, j], 2*t2)
    res = log_map_sphere(p_prime, X[:, N])
    return res

