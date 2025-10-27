# utils_poisson.py
import numpy as np
import h5py
from pyDOE import lhs
import pandas as pd

def read_h5_file(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        f_all = np.array(hf['force_fields'])        # (N_total, nx)
        x_all = np.array(hf['coordinates'])        # (N_total, nx)
        u_all = np.array(hf['solutions'])          # (N_total, nx)
        bc_all = np.array(hf['boundary_conditions'])  # (N_total, 2)
    return f_all, x_all, u_all, bc_all

def get_train_data(h5_path, domain_samples, seq_len, indices, val_indices, sensor_data_dir=None, seed=1234):
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, 'r') as hf:
        f_all = np.array(hf['force_fields'])        # (N_total, nx)
        xdisc = np.array(hf['coordinates'])        # 64 points where discretization is done
        u_all = np.array(hf['solutions'])          # (N_total, nx)
        bc_all = np.array(hf['boundary_conditions'])  # (N_total, 2)
    print(xdisc)
    X = np.repeat(np.expand_dims(xdisc, axis=0), len(indices), axis=0)
    x_left = X[:, 0:1]
    x_right = X[:, -1:]
    umin = np.min(u_all[0])
    umax = np.max(u_all[0])
    us = u_all[indices]
    u_left = np.expand_dims(bc_all[indices, 0:1], axis=1)
    u_right = np.expand_dims(bc_all[indices, 1:2], axis=1)

    if sensor_data_dir is None:
        idx_i = np.sort(np.random.choice(len(X[0]), seq_len, replace=False))
    else:
        idx_i = pd.read_csv(sensor_data_dir)['sensor_indices'].values.reshape(-1)

    xd = X
    fd = f_all[indices]

    xb = np.concatenate((x_left, x_right), axis=1)
    ub = np.concatenate((u_left, u_right), axis=1)

    ins = xd.shape[1]
    bs = xb.shape[1]

    x_sensor = np.repeat(X[0, idx_i].reshape((1, 1, -1)), len(indices), axis=0)
    xbc_in = np.repeat(x_sensor, ins, axis=1).reshape((-1, seq_len))
    xbc_b = np.repeat(x_sensor, bs, axis=1).reshape((-1, seq_len))


    u_sensor = us[:, idx_i].reshape((-1, seq_len, 1))
    ubc_in = np.repeat(u_sensor, ins, axis=1).reshape((-1, seq_len))
    ubc_b = np.repeat(u_sensor, bs, axis=1).reshape((-1, seq_len))

    # validation data
    XeV = np.repeat(np.expand_dims(xdisc, axis=0), len(val_indices), axis=0)
    us_val = u_all[val_indices]

    xv_sens = np.repeat(np.expand_dims(XeV[:, idx_i], axis=1), ins, axis=1).reshape((-1, seq_len))
    uv_sens = us_val[:, idx_i]
    xval = XeV.reshape(len(val_indices), -1, 1)
    uval = us_val.reshape(len(val_indices), -1, 1)
    vals = xval.shape[1]


    xbc_val = np.repeat(xv_sens, vals, axis=1).reshape((-1, seq_len))
    ubc_val = np.repeat(uv_sens, vals, axis=1).reshape((-1, seq_len))

    ivals = {
        'xin': xd.reshape(-1, 1),
        'fin': fd.reshape(-1, 1),
        'xbc': xbc_in,
        'ubc': ubc_in,
        'xbc_b': xbc_b,
        'ubc_b': ubc_b,
        'xb': xb.reshape(-1, 1),
        'fb': np.tile(fd[:, 0], bs).reshape(-1, 1),
        'xval': xval.reshape(-1, 1),
        'fv': np.tile(f_all[val_indices][:, 0], vals).reshape(-1, 1),
        'xbc_val': xbc_val,
        'ubc_val': ubc_val,
    }

    ovals = {
        'ub': ub.reshape(-1,1),
        'uval': uval.reshape(-1,1),
    }

    return ivals, ovals, idx_i, umin, umax