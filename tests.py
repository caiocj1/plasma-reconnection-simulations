import reconnection as r
psi_0, j_0, eta, nu = r.Reconnection.std_input(50, 50)
test = r.Reconnection(psi_0, j_0, eta, nu)
test.run(500, 1e-1)

test.linfit_dpsi_center(1, 30)
