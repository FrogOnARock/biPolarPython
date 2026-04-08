import numpy as np

def make_matrix(d_model: int, d_proj: int) -> list[list[float]]:
    return np.random.normal(0, 1, (d_model, d_proj)).tolist()

def project(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:

    seq_rows, seq_cols = np.shape(np.array(X))
    proj, seq_cols_p = np.shape(np.array(W))

    if seq_cols != seq_cols_p:
        raise ValueError("The number of columns in the token embedding must match the number of columns in the projection matrix.")


    out = [[0] * proj for _ in range(seq_rows)]

    for i in range(seq_rows):
        for j in range(proj):
            out[i][j] = dot_product(X[i], W[j])

    return out

def dot_product(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def compute_similarity(X: list[list[float]], Y: list[list[float]]) -> list[list[float]]:
    sim_matrix = [[0] * len(Y) for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            sim_matrix[i][j] = dot_product(X[i], Y[j])

    return sim_matrix

def asymmetric_check(matrix: list[list[float]]) -> bool:

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                if matrix[i][j] == matrix[j][i]:
                    return False
            else:
                continue
    return True



def main():
    input_matrix = make_matrix(3, 5)
    _, cols = np.shape(np.array(input_matrix))
    W_Q = make_matrix(2, cols)
    W_K = make_matrix(2, cols)

    Q = project(input_matrix, W_Q)
    K = project(input_matrix, W_K)

    np.set_printoptions(precision=3, suppress=True)
    print(f"Q matrix projection: \n{np.array(Q)}")
    print(f"K matrix projection: \n{np.array(K)}\n\n")

    sim_matrix = compute_similarity(Q, K)
    print(f"Similarity matrix: \n{np.array(sim_matrix)}")

    Q_2 = project(input_matrix, W_Q)
    K_2 = project(input_matrix, W_Q)
    sym_sim_matrix = compute_similarity(Q_2, K_2)

    print(f"Symmetrical similarity matrix: \n{np.array(sym_sim_matrix)}")
    print(f"\nOriginal matrix assymetric: {asymmetric_check(sim_matrix)}")
    print(f"\nSymmetric matrix: {asymmetric_check(sym_sim_matrix)}")


if __name__ == "__main__":
   main()
