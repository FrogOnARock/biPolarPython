import numpy as np

def mat_mul(matrix: list[list[int]], vector: list[int]) -> list[int]:

    if len(matrix[0]) != len(vector):
        raise ValueError("Matrix must have the same number of columns as the vector has components.")

    v_r = [0] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            v_r[i] += matrix[i][j] * vector[j]

    return v_r



matrix = [[1, 2], [3, 4], [3, 5]]
vector = [5, 6]
print(mat_mul(matrix, vector))

A = np.array(matrix)
x = np.array(vector)

y = A @ x
print(y)

