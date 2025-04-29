#!/usr/bin/env python3
import sys
from collections import defaultdict

current_key = None
a_vals = defaultdict(float)
b_vals = defaultdict(float)

def emit_result(key, a_vals, b_vals):
    total = 0
    for k in a_vals:
        if k in b_vals:
            total += a_vals[k] * b_vals[k]
    print(f"{key}\t{total}")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    key, val = line.split("\t")
    if key != current_key and current_key is not None:
        emit_result(current_key, a_vals, b_vals)
        a_vals.clear()
        b_vals.clear()

    current_key = key
    tag, k, v = val.split(",")
    k = int(k)
    v = float(v)

    if tag == "A":
        a_vals[k] = v
    elif tag == "B":
        b_vals[k] = v

if current_key is not None:
    emit_result(current_key, a_vals, b_vals)
