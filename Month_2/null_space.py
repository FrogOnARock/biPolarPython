import numpy as np

"""
Point of this here file is to perform a null space check and to determine a method
of finding the null space in each dimension of a collapsed vector


So it's just a linear combination of the n columns of M, where n
    is the number of rows of M

    i.e., if we're projecting 5 dimensions into 2 dimensions, then we will have a
    (2, 5) matrix we're using for the transformation

    Now I guess I should ensure that the I use the first two columns that are linearly
    separable

    So I should probably just check if they're scalar multiples of each other -> the ratio
    between [n][j] / [n][j + 1] and [n+1][j] / [n+1][j + 1] should not be the same. If they
    are not the same then I can say they are linearly separable.

    Following that the null space is just determining the linear combination of the two
    that evaluates to the following vector.

    So whatever gives me c1 + c2 = c3. So need to evaluate c1 + c2 - c3 = 0.

   [1, 0], [0, 1], [2, 2] = 2c1 + 2c2 [2, 2, -1] = 0

   So really it's like whatever scalar multiple that will make the following evaluate to zero

   x * (c1+c2) - c3 = 0
   So in the case above it's
   [1, 0, 2]
   [0, 1, 2]

   or [2, 2]

   Or it could be

   [2, 3, 5]
   [1, 1, 2]

   which would be7
   [1, 1]

   2x + 3y - 5 = 0
   x + 1y - 2 = 0
   2x + 3y - 5 = x + 1y - 2
   x = -2y + 3
   -4y + 6 + 3y = 5
   -1y = -1 y = 1
   x + 1 = 2
   x = 1


   Or

   [2, 1, 9]
   [1, 3, 7]

  [4, 1]

  x2 + y1 - 9 = 0
  x1 + y3 - 7 = 0

  x2 + y1 - 9 = x1 + y3 - 7
  x = 4y + 2

  8y + 4 + y = 9
  9y = 5
  y = 5/9
  x = 5/9 (4) + 2 = 38/9 = 4.22

    So it's a system of equations that I need to solve for.

"""


def null_space(M: list[list[float]], v: list[float]) -> bool:
    return len(M) != len(v)

def find_null_space(M: list[list[float]]) -> list[float]:


    """
    Solving for the null space requires that I perform a linear system of equations.
    For this specific case I'm solving for the null space when transforming an input vector to 2D dimensions.
    This allows me to check for linear dependence by checking if one vector is a scaled copy of another.

    To do this I will iterate through each of the columns in the matrix, and check the
    ratio of current column to following column in the same row, as well as the ratio of current column
    to the following column in the next row. If they are equal, we have linear dependence. Thus we must iterate up one column
    and check again. Once we satisfy our condition we can utilize these columns to solve a linear system of equations
    that allows me to determine the null space for each of the dimensions in excess of our rank.

    [2, 3, 5]

    """

    rows, cols = M.shape

    indexes = None
    for i in range(cols):
        for j in range(i + 1, len(M)):
            if M[i][j] / M[i][i] == M[i+1][j] / M[i+1][i]:
                continue
            else:
                indexes = (i, j)

    i, j = indexes
    A = M[:, [i, j]]

    null_vectors = []
    for k in range(cols):

        if not indexes:
            print("No null space")
            return []

        if k not in indexes:

            coeffs = np.linalg.solve(A, M[:, k])

            nv = np.zeros(cols)
            nv[i] = coeffs[0]
            nv[j] = coeffs[1]
            nv[k] = -1
            null_vectors.append(nv)

    return null_vectors

mats = [[2, 3, 5], [1, 1, 2]]
find_null_space(mats)




