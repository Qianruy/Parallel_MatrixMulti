#!/usr/bin/env python3
import os
import sys
from random import *
import numpy as np

if len(sys.argv) < 4:
    sys.exit(1)

p = int(sys.argv[1])
n = 10
# define matrix A and vector b
A = np.random.rand(n, n)
x = np.random.rand(n)
b = np.dot(A, x)

in_fname1 = f'input_mat_{n}.txt'
in_fname2 = f'input_vec_{n}.txt'
with open(in_fname1, 'w') as f:
        f.write(f'{n}\n')
        for i in range(n):
            f.write(' '.join(map(str, A[i, :])) + '\n')

with open(in_fname2, 'w') as f:
    f.write(' '.join(map(str, b)) + '\n')

out_fname = f'output_{p}.txt'

os.system(f'make && mpirun -np {p} ./pjacobi {in_fname1} {in_fname2} {out_fname}> /dev/null')

with open(out_fname, 'r') as f:
    prog_out = list(map(float, f.readline().strip().split()))

# x_actual = np.linalg.solve(A, b)
# compare prog out with actual x
if np.allclose(np.array(prog_out), x, rtol=1e-14, atol=1e-14):
    print('The program output is correct!')
else:
    print('The program output is incorrect.')

# os.remove(in_fname1)
# os.remove(in_fname2)
# os.remove(out_fname)

