import scipy.linalg
import numpy as np

# Implémentation des opérateurs différentiels
# input  : Matrice (nx + 2, ny + 2)
# output : Matrice (nx, ny)
def Dx(s, dx):
    return (s[:-2, 1:-1] - s[2:, 1:-1])/(dx*2.)

def Dy(s, dy):
    return (s[1:-1, :-2] - s[1:-1, 2:])/(dy*2.)
    
def D2x(s, dx):
    return (s[:-2, 1:-1] - 2.0 * s[1:-1, 1:-1] + s[2:, 1:-1])/(dx**2)
    
def D2y(s, dy):
    return (s[1:-1, :-2] - 2.0 * s[1:-1, 1:-1] + s[1:-1, 2:])/(dy**2)

def CrochetPoisson(f, g, dx, dy):
    return Dx(f, dx)*Dy(g, dy) - Dy(f, dy)*Dx(g, dx)
    
def Lap(s, dx, dy):
    return D2x(s, dx) + D2y(s, dy)

# Opérateur laplacien inverse
# input  : Matrice (nx + 2, ny + 2)
# output : Matrice (nx + 2, ny + 2)
def LapInv(omega, dx, dy):
    nx_bord, ny_bord = omega.shape
    nx, ny = nx_bord - 2, ny_bord - 2

    MLap = np.zeros((nx + 2, nx + 2))
    for i in range(1, nx + 1):
        MLap[i,i] = -2
        MLap[i,i+1] = MLap[i,i-1] = 1
    MLap[0,0] = -2
    MLap[0,1] = 1
    MLap[nx+1,nx+1] = -2
    MLap[nx+1,nx] = 1

    MLap2 = np.zeros((ny + 2, ny + 2))
    for i in range(1, ny + 1):
        MLap2[i,i] = -2
        MLap2[i,i+1] = MLap2[i,i-1] = 1
    MLap2[0,0] = -2
    MLap2[0,1] = 1
    MLap2[ny+1,ny+1] = -2
    MLap2[ny+1,ny] = 1

    return scipy.linalg.solve_sylvester(MLap/(dx**2), MLap2/(dy**2), omega)