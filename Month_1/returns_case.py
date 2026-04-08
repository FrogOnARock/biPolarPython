import numpy as np
import math

def _generate_synthetic_data(price_a: float, price_b: float) -> tuple:

    returns_array = np.random.uniform(low=0.001, high=0.01, size=(20,))
    price_a_array = price_a * np.cumprod(1+ returns_array)
    price_b_array = price_b * np.cumprod(1 + returns_array)

    price_a_returns_array = np.diff(np.log(price_a_array))
    price_b_returns_array = np.diff(np.log(price_b_array))

    return price_a_array, price_b_array, price_a_returns_array, price_b_returns_array

def dot(a, b):
    return sum((x * y for x, y in zip(a, b)))

def norm(a):
    return math.sqrt(dot(a, a))

def cosine_similarity(a, b):
    return dot(a, b)/( norm(a) * norm(b))

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def main():
    v_a, v_b, v_r_a, v_r_b = _generate_synthetic_data(10, 1000)
    print(cosine_similarity(v_r_a, v_r_b))
    print(euclidean_distance(v_r_a, v_r_b))

if __name__ == '__main__':
    main()

