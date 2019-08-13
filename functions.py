def linear(objectx, objecty, objectey):
    '''
    Returns (b, m) values from y = mx + b linear regression.
    Parameters
    ----------
    objectx : array_like
        Independent variable, usually labeled as the x values.
    objecty : array_like
        Dependent variable, usually labeled as the y values.
    objectey : array_like
        Gaussian uncertainties in the y direction.
    Returns
    -------
    (b, eb, m, em) : scalars
        There are two best-fit and their respective standard uncertainty
        variances. The values are returned in a set.
    '''
    import numpy as np
    from numpy.linalg import inv

    # Create matrices and solve the best-fit values.
    Y = objecty
    A = np.matrix([np.ones_like(Y), objectx]).T
    C = np.diag(pow(objectey, 2))
    X1 = inv(A.T @ inv(C) @ A)
    X2 = A.T @ inv(C) @ Y
    X =  X2 @ X1
    return (X.item(0), np.sqrt(X1.item(0)), X.item(1), np.sqrt(X1.item(3)))