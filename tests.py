import reconnection as r
import numpy as np
import warnings

warnings.filterwarnings('ignore')

nx = 50
ny = 50

eta = 0.001
nu = 0.0001

psi_0, j_0 = r.Reconnection.std_input(nx, ny)

test = r.Reconnection(psi_0, j_0, eta, nu)

# ---------------------------

Niter = 2001
dt = 1e-2

test.run(Niter, 1e-2)

# ---------------------------

print(test.psi_hist[:,:,0])
print(test.psi_hist[:,:,Niter-1])

test.plot_psi_center()
test.plot_phi_sheet()
test.plot_psi_sheet()