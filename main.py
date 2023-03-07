import numpy as np



####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    base_functions = []
    polynomial = np.poly1d(0)

    # TODO: Generate Lagrange base polynomials and interpolation polynomial

    for i in range(x.size):
        f = y[i]
        base = np.poly1d([f])
        for j in range(x.size):
            if i != j:
                num = np.poly1d([x[j]], True)
                den = x[i]-x[j]
                base = base * (num / den)
        polynomial += base
        base_functions.append(base/y[i])

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []
    # TODO compute piecewise interpolating cubic polynomials
    for i in range(y.size-1):
        x0 = x[i]
        x1 = x[i+1]
        f0 = y[i]
        f1 = y[i+1]
        fp0 = yp[i]
        fp1 = yp[i+1]
        M = np.array([[1, x0, x0**2, x0**3],
                      [1, x1, x1**2, x1**3],
                      [0, 1, 2*x0, 3*x0**2],
                      [0, 1, 2*x1, 3*x1**2]])

        f = np.array([f0, f1, fp0, fp1])
        a = np.linalg.solve(M, f)

        poly = np.poly1d([a[3], a[2], a[1], a[0]])

        spline.append(poly)

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    n = x.size
    M = np.zeros((4*n-4, 4*n-4))
    M[0][2] = 2
    M[0][3] = 6*x[0]
    M[M.shape[0]-1][M.shape[0]-1] = 6*x[x.size-1]
    M[M.shape[0]-1][M.shape[0]-2] = 2
    xwert = 0
    extra = 1
    for i in range(1, M.shape[0]-3):
        fun = ((i-1)//4)*4
        if extra == 1:
            M[i][fun] = 1
            M[i][fun + 1] = x[xwert]
            M[i][fun + 2] = x[xwert]**2
            M[i][fun + 3] = x[xwert]**3
            xwert += 1

        if extra == 2:
            M[i][fun] = 1
            M[i][fun + 1] = x[xwert]
            M[i][fun + 2] = x[xwert]**2
            M[i][fun + 3] = x[xwert]**3

        if extra == 3:
            M[i][fun] = 0
            M[i][fun + 1] = 1
            M[i][fun + 2] = 2 * x[xwert]
            M[i][fun + 3] = 3 * (x[xwert] ** 2)
            M[i][fun + 4] = 0
            M[i][fun + 5] = -1
            M[i][fun + 6] = -2 * x[xwert]
            M[i][fun + 7] = -3 * (x[xwert] ** 2)

        if extra == 4:
            M[i][fun] = 0
            M[i][fun + 1] = 0
            M[i][fun + 2] = 2
            M[i][fun + 3] = 6 * x[xwert]
            M[i][fun + 4] = 0
            M[i][fun + 5] = 0
            M[i][fun + 6] = -2
            M[i][fun + 7] = -6 * x[xwert]

        if extra == 4:
            extra = 1
        else:
            extra += 1
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4] = 1
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 1] = x[xwert]
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 2] = x[xwert] ** 2
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 3] = x[xwert] ** 3
    xwert += 1

    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4] = 1
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 1] = x[xwert]
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 2] = x[xwert] ** 2
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 3] = x[xwert] ** 3

    f = np.zeros(4 * n - 4)
    ywert = 0
    for j in range(1, f.size, 4):
        f[j] = y[ywert]
        f[j + 1] = y[ywert + 1]
        ywert += 1
    # TODO solve linear system for the coefficients of the spline
    result = np.linalg.solve(M, f)
    spline = []
    # TODO extract local interpolation coefficients from solution
    for k in range(0, M.shape[0], 4):
        poly = np.poly1d([result[k + 3], result[k + 2], result[k + 1], result[k]])
        spline.append(poly)
    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions
    n = x.size
    M = np.zeros((4 * n - 4, 4 * n - 4))
    M[0][1] = 1
    M[0][2] = 2 * x[0]
    M[0][3] = 3 * (x[0] ** 2)
    M[0][M.shape[0] - 3] = - 1
    M[0][M.shape[0] - 2] = - 2 * x[x.size-1]
    M[0][M.shape[0] - 1] = - 3 * (x[x.size-1] ** 2)

    M[M.shape[0] - 1][2] = 2
    M[M.shape[0] - 1][3] = 6 * x[0]
    M[M.shape[0] - 1][M.shape[0] - 1] = - 6 * x[x.size - 1]
    M[M.shape[0] - 1][M.shape[0] - 2] = - 2
    xwert = 0
    extra = 1
    for i in range(1, M.shape[0] - 3):
        fun = ((i - 1) // 4) * 4
        if extra == 1:
            M[i][fun] = 1
            M[i][fun + 1] = x[xwert]
            M[i][fun + 2] = x[xwert] ** 2
            M[i][fun + 3] = x[xwert] ** 3
            xwert += 1

        if extra == 2:
            M[i][fun] = 1
            M[i][fun + 1] = x[xwert]
            M[i][fun + 2] = x[xwert] ** 2
            M[i][fun + 3] = x[xwert] ** 3

        if extra == 3:
            M[i][fun] = 0
            M[i][fun + 1] = 1
            M[i][fun + 2] = 2 * x[xwert]
            M[i][fun + 3] = 3 * (x[xwert] ** 2)
            M[i][fun + 4] = 0
            M[i][fun + 5] = -1
            M[i][fun + 6] = -2 * x[xwert]
            M[i][fun + 7] = -3 * (x[xwert] ** 2)

        if extra == 4:
            M[i][fun] = 0
            M[i][fun + 1] = 0
            M[i][fun + 2] = 2
            M[i][fun + 3] = 6 * x[xwert]
            M[i][fun + 4] = 0
            M[i][fun + 5] = 0
            M[i][fun + 6] = -2
            M[i][fun + 7] = -6 * x[xwert]

        if extra == 4:
            extra = 1
        else:
            extra += 1
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4] = 1
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 1] = x[xwert]
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 2] = x[xwert] ** 2
    M[M.shape[0] - 3][((M.shape[0] - 2) // 4) * 4 + 3] = x[xwert] ** 3
    xwert += 1

    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4] = 1
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 1] = x[xwert]
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 2] = x[xwert] ** 2
    M[M.shape[0] - 2][((M.shape[0] - 1) // 4) * 4 + 3] = x[xwert] ** 3

    f = np.zeros(4 * n - 4)
    ywert = 0
    for i in range(1, f.size, 4):
        f[i] = y[ywert]
        f[i + 1] = y[ywert + 1]
        ywert += 1

    # TODO solve linear system for the coefficients of the spline
    result = np.linalg.solve(M, f)

    spline = []
    # TODO extract local interpolation coefficients from solution
    for i in range(0, M.shape[0], 4):
        poly = np.poly1d([result[i + 3], result[i + 2], result[i + 1], result[i]])
        spline.append(poly)

    return spline


if __name__ == '__main__':

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")


