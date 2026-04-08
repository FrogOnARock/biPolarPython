
import numpy as np
from sympy.codegen.ast import none


# class CoordinateVector:
#     def __init__(self, values: list[list[float]]):
#         self.values = values
#         self.shape: tuple = (len(values), len(values[0]))
#
#
# class Basis:
#     def __init__(self, matrix: list[list[float]]):
#         self.matrix = matrix
#         self.shape: tuple = (len(matrix), len(matrix[0]))
#
#     def _check_square(self):
#         if len(self.matrix) != len(self.matrix[0]):
#             raise ValueError("Matrix must be square.")
#
#     def _check_rank(self):
#         test_vec = [1 * self.shape[0]]
#         test_output = [0 * len(test_vec)]
#         for x in range(self.shape[0]):
#             for y in range(self.shape[1]):
#                 test_output[x] += self.matrix[x][y] * test_vec[y]
#
#
#
#
#         return test_output ==
#
#
#
# class BatchVector:
#     def __init__(self, values: list[list[float]]):
#         self.values = values
#         self.shape = (len(values), len(values[0]))
#
#     def apply_basis(self, matrix: Basis) -> CoordinateVector:
#
#


def make_rank(vec1: list[float], vec2: list[float]) -> list[list[float]]:

    u = len(vec1)
    v = len(vec2)
    out = [[0] * v for _ in range(u)]

    for i in range(u):
        for j in range(v):
            out[i][j] = vec1[i] * vec2[j]

    return out


def matvec(M: list[list[float]], v: list[float]) -> list[float]:

    out_vec = [0] * len(M)

    for i in range(len(M)):
        for j in range(len(v)):
            out_vec[i] += M[i][j] * v[j]

    return out_vec

def is_rank_deficient(M: list[list[float]]) -> bool:

    # I want to check if the matrix is rank deficient
    # I do this by checking the ratio of the rows
    # I need just the ratio of the first element of the rows, so I'd need [i][0] and [i+1][0]
    # When I move up a row I want to check [i][0] and [i+1+1][0]
    # These ratios are what I need to use to examine [i][1] and [i+1+1][1]

    for i in range(len(M)):
        for j in range(i+1, len(M)):
            for u in range(len(M[j])):
                if M[j][u] != 0:
                    k = M[i][u] / M[j][u] #find the first elment in row j that isn't empty
                    break

            for u in range(len(M[j])):
                if M[j][u] == 0:
                    if M[i][u] != 0:
                        return False
                    continue

                if M[i][u] / M[j][u] != k: #check to see if the ratio is the same for each column u
                    return False
                else:
                    continue

    return True


vec1 = [1, 2, 3] # columnar vector
vec2 = [4, 5] # row vector
v1 = [1, 0]
v2 = [0, 1]
v3 = [3, -1]
out = make_rank(vec1, vec2)
out2, out3, out4 = matvec(out, v1), matvec(out, v2), matvec(out, v3)

print(out)
print(out2, out3, out4, sep='\n')
print(is_rank_deficient(out))

# Case 1: rank deficient
M1 = [[1, 2, 3],
      [2, 4, 6]]

# Case 2: full rank
M2 = [[1, 2, 3],
      [0, 1, 0]]

# Case 3: zero guard trigger
M3 = [[1, 0, 3],
      [2, 0, 6]]

print(is_rank_deficient(M1), is_rank_deficient(M2), is_rank_deficient(M3))