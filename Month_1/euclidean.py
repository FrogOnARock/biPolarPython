import math

def euclidean(a, b):

    if len(a) != len(b):
        raise ValueError('vectors must be same length')

    return math.sqrt(sum((x-y) ** 2 for x, y in zip(a, b)))

if __name__ == '__main__':
    print(euclidean([1, 2, 3], [4, 5, 6]))
