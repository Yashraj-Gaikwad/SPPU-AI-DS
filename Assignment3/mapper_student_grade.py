#!/usr/bin/env python3
import sys

# Input: student_id,subject,marks
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("student_id"):
        continue
    student_id, subject, marks = line.split(",")
    print(f"{student_id}\t{marks}")
