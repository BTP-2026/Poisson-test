from pde import CartesianGrid, ScalarField, solve_poisson_equation
import h5py as h5
import numpy as np
import sys

num_samples = int(sys.argv[1])

file_name = f"data/poisson_{num_samples}.h5"

f = np.zeros((num_samples, 64))
xs = np.zeros((num_samples, 64))
u = np.zeros((num_samples, 64))
bc = np.zeros((num_samples, 2))

for i in range(num_samples):
    grid = CartesianGrid([[0,1]], 64)
    x = grid.cell_coords[:,0]
    k = np.random.randint(1, 5)
    a = np.random.uniform(-10, 10)
    data = a * np.sin(k * np.pi * x)
    field = ScalarField(grid, data)
    f[i] = data
    xs[i] = grid.cell_coords[:,0]
    bc0 = {'value': np.random.rand() * 1e1}
    bc[i,0] = bc0['value']
    bc1 = {'value': np.random.rand() * 1e1}
    bc[i,1] = bc1['value']
    solution = solve_poisson_equation(field, bc=[bc0, bc1])
    u[i] = solution.data

# Save to HDF5
with h5.File(file_name, 'w') as hf:
    hf.create_dataset('force_fields', data=f)
    hf.create_dataset('coordinates', data=xs)
    hf.create_dataset('solutions', data=u)
    hf.create_dataset('boundary_conditions', data=bc)

print(f"Dataset saved to {file_name}")



