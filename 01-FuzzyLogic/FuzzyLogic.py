import numpy as np

def union(A, B):
    return np.maximum(A, B)

def intersection(A, B):
    return np.minimum(A, B)

def complement(A):
    return 1 - A

def difference(A, B):
    return np.maximum(A - B, 0)

def cartesian_product(A, B):
    return np.outer(A, B)

def max_min_composition(R1, R2):
    # Max-min composition is usually defined as:
    # C[i][j] = max_k (min(R1[i][k], R2[k][j]))
    # The following implementation uses broadcasting for efficiency.
    n, m = R1.shape
    m2, p = R2.shape
    assert m == m2, "Inner dimensions must match for max-min composition."
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            C[i, j] = np.max(np.minimum(R1[i, :], R2[:, j]))
    return C

A = np.array([0.2, 0.4, 0.7, 0.8])
B = np.array([0.1, 0.8, 0.2, 0.3])

R1 = cartesian_product(A, B)
R2 = cartesian_product(B, A)
result = max_min_composition(R1, R2)

print("Union of A and B:", union(A, B))
print("Intersection of A and B:", intersection(A, B))
print("Complement of A:", complement(A))
print("Difference of A and B:", difference(A, B))
print(f"Cartesian product of A and B:\n{R1}")
print(f"Cartesian product of B and A:\n{R2}")
print(f"Max-min composition of R1 and R2:\n{result}")
