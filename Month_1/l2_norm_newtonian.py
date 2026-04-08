def sqrt_newton(x, tol=1e-8):
    if x < 0:
        raise ValueError('x must be non-negative')

    if x == 0:
        return 0

    s = x
    while True:
        next_s = (s + x / s) / 2
        if abs(next_s - s) < tol:
            return next_s
        s = next_s

def dot(a, b):
    if len(a) != len(b):
        raise ValueError('vectors must be same length')

    total = 0
    for x, y in zip(a, b):
        total += x * y

    return total

def l2_norm(a):
    return sqrt_newton(dot(a, a))

def cosine_similarity(a, b):
    denom = l2_norm(a) * l2_norm(b)
    if denom == 0:
        raise ValueError("Cosine undefined for zero vector")
    return dot(a, b)/denom



if __name__ == '__main__':
    a = [1, 2, 3]
    b = [4, 5, 6]

    print(cosine_similarity(a,b))