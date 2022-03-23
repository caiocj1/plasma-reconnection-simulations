import numpy as np

def bc_psi_const(psi_part, psi_0):
    """
    Adds boundary conditions to psi:
    Value remains constant and equal to psi_0
    @param  : psi_part (nx, ny)
    @param  : psi_0    (nx + 2, ny + 2)
    @return : psi      (nx + 2, ny + 2)
    """
    nx, ny = psi_part.shape
    psi = np.zeros((nx + 2, ny + 2))
    psi[1:-1, 1:-1] = psi_part[:,:]
    psi[0,:] = psi_0[0, :]
    psi[-1,:] = psi_0[-1, :]
    psi[:,0] = psi_0[:, 0]
    psi[:,-1] = psi_0[:, -1]
    return psi

def bc_j_const(j_part, j_0):
    """
    Adds boundary conditions to j:
    Value remains constant and equal to j_0
    @param  : j_part (nx, ny)
    @param  : j_0    (nx + 2, ny + 2)
    @return : j      (nx + 2, ny + 2)
    """
    nx, ny = j_part.shape
    j = np.zeros((nx + 2, ny + 2))
    j[1:-1, 1:-1] = j_part[:,:]
    j[0,:] = j_0[0, :]
    j[-1,:] = j_0[-1, :]
    j[:,0] = j_0[:, 0]
    j[:,-1] = j_0[:, -1]
    return j

def bc_omega_zero(omega_part):
    """
    Adds boundary conditions to psi:
    Value remains constant and equal to 0
    @param  : omega_part (nx, ny)
    @return : omega      (nx + 2, ny + 2)
    """
    nx, ny = omega_part.shape
    omega = np.zeros((nx + 2, ny + 2))
    omega[1:-1, 1:-1] = omega_part[:,:]
    return omega