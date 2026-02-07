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
        self.basis_matrix = None

    def create_coord_vector(self, values, matrix):
        self.values = values
        self.shape = (len(values),)
        self.basis_matrix = matrix.matrix


    def invert_coord_vector(self):
        return np.dot(np.linalg.inv(self.basis_matrix), self.values)


class AmbientVector:
    def __init__(self):
        self.values = None
        self.shape: tuple = (None,)
        self.basis_matrix = None

    def create_amb_vector(self, values):
        self.values = values
        self.shape = (len(values),)

    def apply_basis(self, basis_d: Basis):
        total = [0 for _ in range(len(basis_d.matrix))]
        for i in range(len(basis_d.matrix)):
            for j in range(len(basis_d.matrix[0])):
                total[i] += self.values[j] * basis_d.matrix[i][j]

        return total

if __name__ == '__main__':

    # Instantiate objects
    basis = Basis()
    c_vector = CoordinateVector()
    c_vector_2 = CoordinateVector()
    basis_2 = Basis()
    av_1 = AmbientVector()


    # create the coordinate vector
    fixed_vector = np.array([1,
                         2])
    av_1.create_amb_vector(fixed_vector)


    # create the basis matrix
    matrix_v = np.array([[1, 2], [-1, 2]])
    basis.create_basis(matrix_v)
    matrix_v2 = np.array([[3, 5], [-3, 2]])
    basis_2.create_basis(matrix_v2)

    #show basis matrices as well as coord vector
    print(f"Vector being transformed: {av_1.values} \n")
    print(f"Basis matrix 1: \n{basis.matrix}")
    print(f"Basis matrix 2: \n{basis_2.matrix}")


    c_vector.create_coord_vector(av_1.apply_basis(basis), basis)
    c_vector_2.create_coord_vector(av_1.apply_basis(basis_2), basis_2)

    print("\n The vectors in coordinate space are: \n")
    print(f"The vector in coordinate space is: {c_vector.values}")
    print(f"The vector in coordinate space 2 is {c_vector_2.values}")

    ov_1 = c_vector.invert_coord_vector()
    ov_2 = c_vector_2.invert_coord_vector()
    print("\n The inverted vectors are below: \n")
    print(f"Original vector {ov_1}")
    print(f"Original vector {ov_2}")

