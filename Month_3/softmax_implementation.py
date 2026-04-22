import math

def naive_softmax(x: list[float]) -> list[float]:
    total = sum([math.exp(a) for a in x])
    return [(math.exp(y) / total) for y in x]

def stable_softmax(x: list[float]) -> list[float]:
    max_value = max(x)
    total = sum([math.exp(a-max_value) for a in x])
    return [math.exp(y - max_value) / total for y in x]

arr1 = [1e308, 1e308+1, 1e308+2]
arr2 = [0.0, 0.0, 0.0, 0.0, 0.0]
arr3 = [-1e10, -1e10, -1e10]
arr4 = [1000.0, 0.0, 0.0]
arr5 = [1.0, 2.0, 3.0]
arr6 = [1000.0, 1001.0, 1002.0]

print(stable_softmax(arr1))
print(stable_softmax(arr2))
print(stable_softmax(arr3))
print(stable_softmax(arr4))
print(stable_softmax(arr6))
print(stable_softmax(arr5))
