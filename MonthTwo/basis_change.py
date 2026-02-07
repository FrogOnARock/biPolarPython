import numpy as np

class Basis:
    def __init__(self):
        self.matrix = None
        self.shape: tuple = (None, None)

    def create_basis(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape


class CoordinateVector:
    def __init__(self):
        self.values = None
        self.shape: tuple = (None,)

    def create_coord_vector(self, values):
        self.values = values
        self.shape = (len(values),)

    def apply_basis(self, basis_d: Basis):
        total = [0 for _ in range(len(basis_d.matrix))]
        for i in range(len(basis_d.matrix)):
            for j in range(len(basis_d.matrix[0])):
                total[i] += self.values[j] * basis_d.matrix[i][j]

        return total


class AmbientVector:
    def __init__(self):
        self.values = None
        self.shape: tuple = (None,)
        self.basis_matrix = None

    def create_ambient_vector(self, values, matrix):
        self.values = values
        self.shape = (len(values),)
        self.basis_matrix = matrix.matrix

    def invert_ambient_vector(self):
        return np.dot(np.linalg.inv(self.basis_matrix), self.values)


basis = Basis()
coord_vector = CoordinateVector()

# create the coordinate vector
fixed_vector = np.array([1,
                         2])
coord_vector.create_coord_vector(fixed_vector)


# create the basis matrix
matrix_v = np.array([[1, 2], [-1, 2]])
basis.create_basis(matrix_v)
print(coord_vector.values)
print(basis.matrix)

av_1 = AmbientVector()
av_1.create_ambient_vector(coord_vector.apply_basis(basis), basis)
print(av_1.values)

ov_1 = av_1.invert_ambient_vector()
print(ov_1)

