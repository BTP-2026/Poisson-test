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
np.random.seed(13123)
grid = CartesianGrid([[0, 10]], shape=(num_points,))
xs = grid.cell_coords
data = 1
field = ScalarField(grid, data)



import numpy as np
import matplotlib.pyplot as plt


def finite_difference_poisson(n, u_0, u_n, L = 1):

    # Discretize the domain in random locations
    x = np.linspace(0, L, n)
    x[0], x[-1] = 0, L  # domain length

    # Initialize matrix A and vector b with zeros
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Calculate the elements of A and b for each node
    for i in range(1, n - 1):
        A[i, i] = 1 / (x[i] - x[i - 1]) + 1 / (x[i + 1] - x[i])
        A[i, i - 1] = - 1 / (x[i] - x[i - 1])
        A[i, i + 1] = - 1 / (x[i + 1] - x[i])
        b[i] = 0.5 * (x[i + 1] - x[i - 1])
    # Setting entries of the first and the last node
    A[0, 0] = 1 / (x[1] - x[0])
    A[0, 1] = - 1 / (x[1] - x[0])
    A[-1, -1] = 1 / (x[-1] - x[-2])
    A[-1, -2] = - 1 / (x[-1] - x[-2])
    b[0] = 0.5 * (x[1] - x[0])
    b[-1] = 0.5 * (x[-1] - x[-2])

    # Applying boundary conditions
    b[1] += - A[1, 0] * u_0
    b[-2] += - A[-2, -1] * u_n


    u = np.zeros(n)
    u[1:n - 1] = np.dot(np.linalg.inv(A[1:n - 1, 1:n - 1]), b[1:n - 1])
    u[0] = u_0
    u[-1] = u_n

    return x, u


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


bc0 = {'value': 0}
bc1 = {'value': 0}
for i in range(num_samples):
    f[i] = np.ones(num_points)
    bc[i,0] = bc0['value']
    bc[i,1] = bc1['value']
    xs, u[i] = finite_difference_poisson(num_points, bc0['value'], bc1['value'])
    if i % 100 == 0:
        print(f"Generating sample {i}/{num_samples}")
        plt.plot(xs, u[i])
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Solution for sample {i}')
        plt.show()


# Save to HDF5
with h5.File(file_name, 'w') as hf:
    hf.create_dataset('force_fields', data=f)
    hf.create_dataset('coordinates', data=xs)
    hf.create_dataset('solutions', data=u)
    hf.create_dataset('boundary_conditions', data=bc)




# # get_train_data(file_name, domain_samples=1022, seq_len=60, indices=np.arange(80), val_indices=np.arange(80,100))
