#!/usr/bin/env python3
import sys

# Dimensions (hardcoded or passed through env/config)
# A is m x n
# B is n x p
m = 2  # rows in A
n = 2  # cols in A / rows in B
p = 2  # cols in B

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    matrix, i, j, value = line.split(",")
    i = int(i)
    j = int(j)
    value = float(value)

    if matrix == "A":
        for col in range(p):
            # Key: i,col ; Value: A,j,value
            print(f"{i},{col}\tA,{j},{value}")
    elif matrix == "B":
        for row in range(m):
            # Key: row,j ; Value: B,i,value
            print(f"{row},{j}\tB,{i},{value}")
