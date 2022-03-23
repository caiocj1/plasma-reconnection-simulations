import reconnection as r
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm

warnings.filterwarnings('ignore')

nx = 70
ny = 70


psi_0, j_0, eta, nu = r.Reconnection.std_input(nx, ny)

#test = r.Reconnection(psi_0, j_0, eta, nu)

# ---------------------------

Niter = 500
dt = 1e-2

#test.run(Niter, 1e-2)

# ---------------------------

#test.plot_sheet()
#test.linfit_dpsi_center()
#test.plot_dpsi_center()




CenterData = np.zeros([10, Niter])
counter = 0
for etta in np.linspace(0.0001,0.01,10):
    test = r.Reconnection(psi_0, j_0, etta, nu)
    test.run(Niter, 1e-2)
    print("eta = ", etta)
    center_data = test.dpsi_center()
    CenterData[counter,: ]= center_data
    counter += 1


"""
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#CenterData = np.zeros([10, Niter])
#counter = 0
for dtemp in np.linspace(0.05855,0.05875,20):
    test = r.Reconnection(psi_0, j_0, eta, nu)
    NI = int(5 / dtemp)
    test.run(NI, dtemp)
    print("Pas de temps = ", dtemp)
    center_data = test.dpsi_center()
    xdata = np.ones(NI) * dtemp
    ydata = np.arange(NI) * dtemp
    ax.plot(xdata, ydata, np.log10(center_data))
    #CenterData[counter,: ]= center_data
    #counter += 1

"""

Xc, Yc = np.meshgrid(0.01*np.arange(Niter),np.linspace(0.0001,0.01,10))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


surf = ax.plot_surface(Yc, Xc, np.log10(CenterData), cmap=cm.coolwarm, linewidth=0, antialiased=True)
plt.xlabel('$\eta$')
plt.ylabel('$t$')
ax.set_zlabel('$\log(d\Psi)$')


plt.show()
