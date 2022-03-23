import reconnection as r
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

nx = 50
ny = 50

eta = 0.001
nu = 0.0001

psi_0, j_0 = r.Reconnection.std_input(nx, ny)

test = r.Reconnection(psi_0, j_0, eta, nu)

# ---------------------------

Niter = 500
dt = 1e-1

# ---------------------------

test.run(Niter, dt)
test.plot_dpsi_center()
test.plot_sheet()
print(test.linfit_dpsi_center())


