# Plasma Reconnection Simulations
Simulates a current sheet given certain initial conditions and precising boundary conditions,
with normalized units.

Made as part of a PSC (collective scientific project) for École Polytechnique, France.

### Contributors

**PSC Tutor** \
Timothée Nicolas, _PhD_

**Simulation Group** \
Caio Azevedo, _X2020_ \
Tianjao Cao, _X2020_ \
Jean-Baptiste Labit, _X2020_

**Theory Group** \
Domitille Chebat, _X2020_ \
Jianjing Dong, _X2020_ \
Solano Felício, _X2020_

### Instructions

Following is a simple snippet that creates a new current sheet and runs the calculations:

```
import reconnection as r

psi_0, j_0, eta, nu = r.Reconnection.std_input(50, 50)
test = r.Reconnection(psi_0, j_0, eta, nu)
test.run(500, 1e-1)
```

`r.Reconnection.std_input` returns a standard initial condition and certain values for
`eta` and `nu`, which represent the resistivity and viscosity of the plasma, taken as
constant throughout the sheet. It takes as parameters the spatial resolution of the
sheet, and as optional arguments different values for `eta` and `nu`.

`r.Reconnection(psi_0, j_0, eta, nu)` returns the reconnection model with given
initial conditions.

`obj.run(Niter, dt)` runs the simulation with `Niter` iterations of `dt` time-step.

A variety of post-treatment and plotting methods are available.