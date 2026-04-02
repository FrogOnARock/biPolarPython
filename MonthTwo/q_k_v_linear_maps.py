import numpy as np

def create_projection(d_model: int, d_proj: int) -> list[list[float]]:
    """Return a d_proj x d_model projection matrix with small random values"""
    """ Purpose of this is to initialize three matrices for the three different projections: Q, K, V"""
    return np.random.normal(0, 1, (d_proj, d_model)).tolist()


def project(X: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    """
    X: (seq_len, d_model) — each row is a token embedding
    W: (d_proj, d_model) — projection matrix
    Returns: (seq_len, d_proj)

    Must validate shapes explicitly before computing.
    Must raise on mismatch.
    """

    # Need to validate the shapes are as follows, for any rank D input matrix I need at (d_proj, d)
    # So this is working off a row vector, so I need to validate that the length of the row matches the length of

    seq_rows, seq_cols = np.shape(np.array(X))
    d_proj, d_model = np.shape(np.array(W))

    # W_t = [[0] * proj_rows for _ in range(proj_cols)]

    if seq_cols != d_model:
        raise ValueError("The number of columns in the token embedding must match the number of columns in the projection matrix."
                         "(The rows in the transposed projection matrix")

    # for i in range(proj_rows):
    #     for j in range(seq_cols):
    #         W_t[j][i] = W[i][j]
    #
    # d_model, d_proj = np.shape(np.array(W_t))
    # if d_model != seq_cols:
    #     raise ValueError("Incorrect matrix transposition.")
    #

    proj_matrix = [[0] * d_proj for _ in range(seq_rows)]

    # An interesting realization while doing this is that the original matrix, unmodified can be viewed as
    # The transposed matrix row wise. So if I wanted to do the most efficient calculations of dot products of
    # the input vectors row * the transposed matrix's column, I can just leave the matrix as it and take the dot product
    # of row by row and then place it in the appropriate position in the projection. Row 1 by row 1 ends up
    # in row 1 col 1, row 1 by row 2 ends up in row 1 col 2, row 2 by row 1 ends up in row 2 col 1, and row 2 by row 2
    # ends up in row 2 col 2.
    # This the efficient way to do this.

    for i in range(seq_rows):
        for j in range(d_proj):
            proj_matrix[i][j] += dot_product(X[i], W[j])

    return proj_matrix

def dot_product(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors. Must validate equal length."""

    if len(a) != len(b):
        raise ValueError("Vectors must be same length.")

    total = 0
    for i in range(len(a)):
        total += float(a[i]) * float(b[i])
    return total

# Initalize a random input matrix

X = np.random.normal(0, 1, (3, 4))

# D_model is the length of the input matrix, or if it was column wise, the number of columns in the projection matrix
# Once we transpose this matrix, we can take X @ W_(q, k, v)^t. So there should be a match between columns in the
# input vector and the columns in the projection matrix`
d_model = len(X[0])

# Projecting into any space that has rank less than the rank of the input vector
d_proj = d_model - 1

# Initialize the projection matrices
W_q = create_projection(d_model, d_proj)
W_k = create_projection(d_model, d_proj)
W_v = create_projection(d_model, d_proj)

X_q = project(X, W_q)
X_k = project(X, W_k)
X_v = project(X, W_v)


dot_ps = [[0] * len(X_q) for _ in range(len(X_k))]
for i in range(len(X_q)):
     for j in range(len(X_k)):
        dot_ps[i][j] = dot_product(X_q[i], X_k[j])

print(f"Similarity of the projected query and key embeddings is: \n{np.array(dot_ps)}")

dot_p_ov = [[0] * len(X) for _ in range(len(X))]
for i in range(len(X)):
    for j in range(len(X)):
        dot_p_ov[i][j] = dot_product(X[i], X[j])

print(f"Similarity of the original vector with itself is: \n{np.array(dot_p_ov)}")
print("\nPatterns do not really hold when comparing the projected similarity to the original vectors similarity with itself.\n"
      "This is a clear indication of the differences in how projection, and in particular intentional projection, can change the meaning"
      "of the vector we're dealing with, and allow us to manipulate a vector so that we can compare it's projections in a lower"
      "dimensional space.")