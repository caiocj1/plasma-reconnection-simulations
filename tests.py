import reconnection as r
psi_0, j_0, eta, nu = r.Reconnection.std_input(80, 80)
test = r.Reconnection(psi_0, j_0, eta, nu)
test.run(6000, 1e-2)
test.plot_sheet()
test.plot_mag_field(5999)
