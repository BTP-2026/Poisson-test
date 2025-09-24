# utils_poisson.py
import numpy as np
import h5py

def get_train_data(h5_path, domain_samples, seq_len, indices, val_indices, seed=1234):
    rng = np.random.RandomState(seed)

    with h5py.File(h5_path, 'r') as hf:
        f_all = np.array(hf['force_fields'])        # (N_total, nx)
        x_all = np.array(hf['coordinates'])        # (N_total, nx)
        u_all = np.array(hf['solutions'])          # (N_total, nx)
        bc_all = np.array(hf['boundary_conditions'])  # (N_total, 2)

    n_train = len(indices)
    n_val = len(val_indices)
    nx = x_all.shape[1]

    f_train = f_all[indices]
    x_train = x_all[indices]
    u_train = u_all[indices]
    bc_train = bc_all[indices]

    f_val = f_all[val_indices]
    x_val = x_all[val_indices]
    u_val = u_all[val_indices]
    bc_val = bc_all[val_indices]

    idxs = rng.randint(0, nx, size=(n_train, domain_samples))

    xd = x_train[np.arange(n_train)[:, None], idxs].reshape(-1, 1).astype(np.float32)
    fd = f_train[np.arange(n_train)[:, None], idxs].reshape(-1, 1).astype(np.float32)

    # boudry points xb is just drichlet bc at 0 and 1 for all samples see solver.py for more
    xb = np.tile(np.array([[0.0], [1.0]], dtype=np.float32), (n_train, 1))  
    ub = bc_train.astype(np.float32).reshape(-1, 1)
    fb = f_train[:, [0, -1]].astype(np.float32).reshape(-1, 1)                

   
    idx_si = rng.choice(nx, seq_len, replace=False)
    x_sens = x_train[:, idx_si].reshape(n_train, 1, seq_len).astype(np.float32)   # (n_train,1,seq_len)
    u_sens = u_train[:, idx_si].reshape(n_train, 1, seq_len).astype(np.float32)
    f_sens = f_train[:, idx_si].reshape(n_train, 1, seq_len).astype(np.float32)

    def expand_for(n_repeat, sens):   # sens shape (n_train, 1, seq_len) -> returns (n_train * n_repeat, seq_len, 1)
        out = np.repeat(sens, n_repeat, axis=1)               # (n_train, n_repeat, seq_len)
        out = out.reshape(-1, seq_len)                        # (n_train*n_repeat, seq_len)
        out = out.reshape(-1, seq_len, 1)                     # (n_train*n_repeat, seq_len, 1)
        return out.astype(np.float32)

    ins = domain_samples
    bs = 2

    xbc_in = expand_for(ins, x_sens)
    ubc_in = expand_for(ins, u_sens)
    fbc_in = expand_for(ins, f_sens)

    xbc_b = expand_for(bs, x_sens)     
    ubc_b = expand_for(bs, u_sens)
    fbc_b = expand_for(bs, f_sens)


    xval = x_val.reshape(-1, 1).astype(np.float32)
    uval = u_val.reshape(-1, 1).astype(np.float32)
    fval = f_val.reshape(-1, 1).astype(np.float32)

    vals = xval.shape[0] // n_val 
    xv_sens = x_val[:, idx_si].reshape(n_val, 1, seq_len).astype(np.float32)
    uv_sens = u_val[:, idx_si].reshape(n_val, 1, seq_len).astype(np.float32)
    fv_sens = f_val[:, idx_si].reshape(n_val, 1, seq_len).astype(np.float32)

    xbc_val = expand_for(vals, xv_sens)
    ubc_val = expand_for(vals, uv_sens)
    fbc_val = expand_for(vals, fv_sens)

    ivals = {
        'xin': xd,
        'fin': fd,
        'xbc_in': xbc_in,
        'fbc_in': fbc_in,
        'ubc_in': ubc_in,
        'xb': xb,
        'fb': fb,
        'xbc_b': xbc_b,
        'fbc_b': fbc_b,
        'ubc_b': ubc_b,
        'xval': xval,
        'fval': fval,
        'xbc_val': xbc_val,
        'fbc_val': fbc_val,
        'ubc_val': ubc_val
    }

    ovals = {
        'ub': ub,
        'uval': uval
    }

    return ivals, ovals, idx_si


# ret = get_train_data_poisson('data/poisson_5000.h5', domain_samples=2000, seq_len=60, indices=np.arange(80), val_indices=np.arange(80, 100))

# for k, v in ret.items():
#     print(k, v.shape)