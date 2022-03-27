import reconnection as r
psi_0, j_0, eta, nu = r.Reconnection.std_input(50, 50)
test = r.Reconnection(psi_0, j_0, eta, nu)
test.run(500, 1e-1)

test.plot_mag_field(500)
print(test.sheet_size(500))
