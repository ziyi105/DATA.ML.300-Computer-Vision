import numpy as np


def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    A = np.vstack([
        x1[0] * x2[0], x1[0] * x2[1], x1[0],
        x1[1] * x2[0], x1[1] * x2[1], x1[1],
        x2[0], x2[1], np.ones_like(x1[0])
    ]).T
    ##-your-code-ends-here-##

    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##
    _, _, vh = np.linalg.svd(A)
    v_min = vh[-1, :]
    ##-your-code-ends-here-##
    F = np.reshape(v_min, (3, 3))  # reshape to acquire Fundamental matrix F

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##
    U, S, Vh = np.linalg.svd(F)
    S[2] = 0 
    F = np.dot(U, np.dot(np.diag(S), Vh))
    ##-your-code-ends-here-##
    
    return F
