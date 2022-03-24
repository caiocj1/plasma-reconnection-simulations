import reconnection as r
psi_0, j_0, eta, nu = r.Reconnection.std_input(50, 50)
test = r.Reconnection(psi_0, j_0, eta, nu)
test.run(5000, 1e-2)

test.plot_dpsi_center()
test.plot_deriv_dpsi_center()
