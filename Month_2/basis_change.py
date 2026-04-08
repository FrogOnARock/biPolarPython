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

if __name__ == '__main__':

    # Instantiate objects
    basis = Basis()
    coord_vector = CoordinateVector()
    basis_2 = Basis()
    av_1 = AmbientVector()
    av_2 = AmbientVector()


    # create the coordinate vector
    fixed_vector = np.array([1,
                         2])
    coord_vector.create_coord_vector(fixed_vector)


    # create the basis matrix
    matrix_v = np.array([[1, 2], [-1, 2]])
    basis.create_basis(matrix_v)
    matrix_v2 = np.array([[3, 5], [-3, 2]])
    basis_2.create_basis(matrix_v2)

    #show basis matrices as well as coord vector
    print(f"Vector being transformed: {coord_vector.values} \n")
    print(f"Basis matrix 1: \n{basis.matrix}")
    print(f"Basis matrix 2: \n{basis_2.matrix}")


    av_1.create_ambient_vector(coord_vector.apply_basis(basis), basis)
    av_2.create_ambient_vector(coord_vector.apply_basis(basis_2), basis_2)

    print("\n The vectors in ambient space are: \n")
    print(f"The vector in ambient space is: {av_1.values}")
    print(f"The vector in ambient space 2 is {av_2.values}")

    ov_1 = av_1.invert_ambient_vector()
    ov_2 = av_2.invert_ambient_vector()
    print("\n The inverted vectors are below: \n")
    print(f"Original vector {ov_1}")
    print(f"Original vector {ov_2}")

    fixed_av = AmbientVector()
    fixed_av.create_ambient_vector(fixed_vector, basis)
    fixed_av2 = AmbientVector()
    fixed_av2.create_ambient_vector(fixed_vector, basis_2)

    print("\n The inverted vectors with a fixed ambient vector and different basis are: \n")
    print(fixed_av.invert_ambient_vector())
    print(fixed_av2.invert_ambient_vector())




