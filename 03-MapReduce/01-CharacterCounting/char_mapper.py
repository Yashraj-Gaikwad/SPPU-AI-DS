# char_mapper.py
import sys

for line in sys.stdin:
    for char in line.strip():
        print(f"{char}\t1")
