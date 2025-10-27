from pde import CartesianGrid, ScalarField, solve_poisson_equation
import h5py as h5
import numpy as np
import sys
from utils import *

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

num_samples = int(sys.argv[1])
num_points = int(sys.argv[2])
file_name = f"data/poisson_{num_samples}.h5"

f = np.zeros((num_samples, num_points))
u = np.zeros((num_samples, num_points))
bc = np.zeros((num_samples, 2))
np.random.seed(1234)
grid = CartesianGrid([[0, 10]], shape=(num_points,))
xs = grid.cell_coords
data = 1
field = ScalarField(grid, data)
bc0 = {'value': 0}
bc1 = {'value': 0}

def solve_fd(N):
    h = 1.0 / N
    # right-hand side (interior points only)
    b = -h**2 * np.ones(N-1)

    # tridiagonal matrix A
    diagonals = [1, -2, 1]                     # sub, main, super
    offsets   = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(N-1, N-1), format='csr')

    # solve Au = b
    u_interior = spsolve(A, b)

    # build full solution including boundaries
    u = np.concatenate(([0.0], u_interior, [0.0]))
    x = np.linspace(0, 1, N+1)
    return x, u


def u_exact(x):
    return x * (1 - x) / 2

for i in range(num_samples):
    if i % 100 == 0:
        print(f"Generating sample {i}/{num_samples}")
    f[i] = data
    bc[i,0] = bc0['value']
    bc[i,1] = bc1['value']
    xs, u[i] = solve_fd(num_points - 1)


# Save to HDF5
with h5.File(file_name, 'w') as hf:
    hf.create_dataset('force_fields', data=f)
    hf.create_dataset('coordinates', data=xs)
    hf.create_dataset('solutions', data=u)
    hf.create_dataset('boundary_conditions', data=bc)


# # get_train_data(file_name, domain_samples=1022, seq_len=60, indices=np.arange(80), val_indices=np.arange(80,100))
