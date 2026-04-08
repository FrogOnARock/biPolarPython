import math

def mean(xs):
    total = sum(i for i in xs)
    return total / len(xs)

def variance(xs):
    n = len(xs)
    mx = mean(xs)

    total = 0.0
    for i in range(n):
        total += (xs[i] - mx)**2

    return total / (n - 1)

def covariance(xs, ys):

    if len(xs) != len(ys):
        raise ValueError("length mismatch")

    n = len(xs)

    if n < 2:
        raise ValueError("need at least two points")


    mx = mean(xs)
    my = mean(ys)

    total = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        total += dx * dy

    return total / (n - 1)

def correlation(xs, ys):

    epsilon = 1e-15
    domain_limit = 1e-16
    denom = math.sqrt(variance(xs) * variance(ys))

    if denom < max(domain_limit, epsilon * denom):
        return ValueError("Division by zero -> exploding noise")


    return covariance(xs, ys) / denom



