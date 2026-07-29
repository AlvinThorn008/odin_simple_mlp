"""
Utility script to help cross check results against numpy

Mostly intended for interactive use i.e.

python -i main.py
"""

import numpy as np
import time

def mat(file_name, cm=False):
    with open('outputs/'+file_name, 'rb') as f:
        rows, cols = np.fromfile(f, dtype=np.uint64, count=2)
        data = np.fromfile(f, dtype=np.float32, count=rows*cols)

    return data.reshape(rows, cols) if not cm else data.reshape(cols, rows).T

a = mat("a.bin")
b = mat("b.bin")
c = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

print("\n[Numpy / OpenBLAS]")

total = 0.0
for i in range(0, 10):
    now = time.time()
    np.matmul(a, b, out=c)
    total += time.time() - now
    c = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

print("10 runs completed")
print("Total time:       " + f"\x1b[96m{total:.3f}s\x1b[0m")
print("Average run time: " + f"\x1b[96m{total/10:.3f}s\x1b[0m")