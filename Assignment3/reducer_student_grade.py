#!/usr/bin/env python3
import sys

def get_grade(avg):
    avg = float(avg)
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    elif avg >= 70:
        return 'C'
    elif avg >= 60:
        return 'D'
    else:
        return 'F'

current_id = None
total = 0
count = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    student_id, marks = line.split("\t")
    marks = float(marks)

    if current_id == student_id:
        total += marks
        count += 1
    else:
        if current_id:
            avg = total / count
            print(f"{current_id}\t{avg:.2f}\t{get_grade(avg)}")
        current_id = student_id
        total = marks
        count = 1

# Output for the last student
if current_id:
    avg = total / count
    print(f"{current_id}\t{avg:.2f}\t{get_grade(avg)}")
