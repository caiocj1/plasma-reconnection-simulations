import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time
import numpy as np
import calc
import bc


class Reconnection:

    def __init__(self, psi_0: np.ndarray, j_0: np.ndarray, eta: float, nu: float) -> None:
        """
        Initializes reconnection model
        @param : nx, ny     | grid resolution
        @param : psi_0, j_0 | initial distributions of psi and j
        @param : eta        | constant current sheet resistivity
        @param : nu         | constant plasma viscosity
        """
        nx_large, ny_large = psi_0.shape
        self.nx, self.ny = nx_large - 2, ny_large - 2

        self.eta = eta
        self.nu = nu

        self.psi_0 = psi_0
        self.j_0 = j_0

        x, self.dx = np.linspace(-1, 1, self.nx + 2, retstep=True)
        y, self.dy = np.linspace(-1, 1, self.ny + 2, retstep=True)

        self.Y, self.X = np.meshgrid(y,x)

    @staticmethod
    def std_input(nx: int, ny: int) -> np.ndarray:
        k = 10.0
        x = np.linspace(-1, 1, nx + 2)
        y = np.linspace(-1, 1, ny + 2)

        Y, X = np.meshgrid(y,x)

        psi_0 = np.log(np.cosh(k*Y))/k
        j_0 = k/np.cosh(k*Y)**2
        
        return psi_0, j_0

    def F(self, X: np.ndarray) -> np.ndarray:
        """
        Evolution operator of psi and omega, forcing
        constant psi and zero omega in boundaries
        @param  : X    (nx, ny, 2) | values of psi in X[:,:,0] and of omega in X[:,:,1]
        @return : F(X) (nx, ny, 2) | new values of psi in F(X)[:,:,0] and of omega in F(X)[:,:,1]
        """
        psi_part = X[:, :, 0]
        psi = bc.bc_psi_const(psi_part, self.psi_0)
        dpsi = psi - self.psi_0
        dj_part = calc.Lap(dpsi, self.dx, self.dy)
        dj = bc.bc_omega_zero(dj_part)

        omega_part = X[:, :, 1]
        omega = bc.bc_omega_zero(omega_part)

        F_new = np.zeros((self.nx, self.ny, 2))

        phi = calc.LapInv(omega, self.dx, self.dy)

        CPhiPsi = calc.CrochetPoisson(phi, psi, self.dx, self.dy)
        CPhiomega = calc.CrochetPoisson(phi, omega, self.dx, self.dy)
        CPsij = calc.CrochetPoisson(
            self.psi_0, dj, self.dx, self.dy) + calc.CrochetPoisson(dpsi, self.j_0, self.dx, self.dy) + calc.CrochetPoisson(dpsi, dj, self.dx, self.dy)

        F_new[:, :, 0] = -CPhiPsi + self.eta * dj[1:-1, 1:-1]
        F_new[:, :, 1] = -CPhiomega + CPsij + self.nu * calc.Lap(omega, self.dx, self.dy)

        return F_new

    def RK4(self, X: np.ndarray, dt: float) -> np.ndarray:
        """
        Classic Runge-Kutta method
        @param  : X          (nx, ny, 2)
        @return : RK4(X, dt) (nx, ny, 2)
        """
        k1 = self.F(X)
        k2 = self.F(X + k1 * dt * 0.5)
        k3 = self.F(X + k2 * dt * 0.5)
        k4 = self.F(X + k3 * dt)
        return dt * (k1 + 2. * k2 + 2. * k3 + k4)/(6.0)

    def RKTimeStep(self, X: np.ndarray, dt: float) -> np.ndarray:
        """
        Time step applying RK4
        @param  : X              (nx, ny, 2)
        @return : X + RK4(X, dt) (nx, ny, 2)
        """
        return X + self.RK4(X, dt)
    
    def run(self, Niter: int, dt: float) -> None:
        """
        Run simulation
        @param : Niter | number of iterations
        @param : dt    | amount of time between iterations
        """
        if Niter < 2 or dt <= 0:
            raise Exception('Invalid values for number of iterations (>= 2) or time step (> 0)')

        start_time = time.time()

        self.t = np.zeros(Niter)
        psi_hist = np.zeros((self.nx, self.ny, Niter))
        A = 1e-7
        dPsi = A*np.cos(2*np.pi*self.X)
        psi_hist[:, :, 0] = (self.psi_0 + dPsi)[1:-1, 1:-1]

        omega_hist = np.zeros((self.nx, self.ny, Niter))
 
        for i in range(Niter-1):
            cur = np.zeros((self.nx, self.ny, 2))
            cur[:, :, 0] = psi_hist[:, :, i]
            cur[:, :, 1] = omega_hist[:, :, i]

            new_x = self.RKTimeStep(cur, dt)
            self.t[i+1] = self.t[i]+dt
            psi_hist[:, :, i+1] = new_x[:, :, 0]
            omega_hist[:, :, i+1] = new_x[:, :, 1]
        
        self.omega_hist = omega_hist
        self.psi_hist = psi_hist

        print('run() done, %0.3f' % (time.time() - start_time), 's')

    def plot_psi_center(self) -> None:
        """
        Plot evolution of psi in center of current sheet
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]
        dpsi = np.zeros(Niter)
        for i in range(Niter):
            dpsi[i] = self.psi_hist[self.nx//2, self.ny//2, i] - self.psi_0[self.nx//2, self.ny//2]
        plt.figure(1)
        plt.clf()
        plt.semilogy(self.t, np.abs(dpsi), lw=2)
        plt.show()
        
    def plot_phi_sheet(self) -> None:
        """
        Plot evolution of phi in whole current sheet
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]

        omega_hist = self.omega_hist
        dx, dy, X, Y = self.dx, self.dy, self.X, self.Y
        t = self.t

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        omega = bc.bc_omega_zero(omega_hist[:, :, 0])
        phi = calc.LapInv(omega, self.dx, self.dy)

        ax.contourf(self.X,self.Y,phi,20,cmap=plt.get_cmap('coolwarm'))
        ax.contour(self.X,self.Y,phi,20,colors='k',alpha=0.2)
        ax.set_title('t = %0.3f' % self.t[0])

        class Index():
            ind = 0

            def next(self, event):
                self.ind += 1
                i = self.ind % Niter
                ax.cla()
                omega = bc.bc_omega_zero(omega_hist[:, :, i])
                phi = calc.LapInv(omega, dx, dy)

                ax.contourf(X,Y,phi,20,cmap=plt.get_cmap('coolwarm'))
                ax.contour(X,Y,phi,20,colors='k',alpha=0.2)
                ax.set_title('t = %0.3f' % t[i])
                plt.draw()

            def prev(self, event):
                self.ind -= 1
                i = self.ind % Niter
                ax.cla()
                omega = bc.bc_omega_zero(omega_hist[:, :, i])
                phi = calc.LapInv(omega, dx, dy)

                ax.contourf(X,Y,phi,20,cmap=plt.get_cmap('coolwarm'))
                ax.contour(X,Y,phi,20,colors='k',alpha=0.2)
                ax.set_title('t = %0.3f' % t[i])
                plt.draw()

        callback = Index()
        axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

        plt.show()
        
    def plot_psi_sheet(self) -> None:
        """
        Plot evolution of phi in whole current sheet
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]

        psi_0 = self.psi_0
        psi_hist = self.psi_hist
        X, Y = self.X, self.Y
        t = self.t

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        psi = bc.bc_psi_const(psi_hist[:, :, 0], psi_0)

        ax.contourf(self.X,self.Y,psi,20,cmap=plt.get_cmap('coolwarm'))
        ax.contour(self.X,self.Y,psi,20,colors='k',alpha=0.2)
        ax.set_title('t = %0.3f' % self.t[0])

        class Index():
            ind = 0

            def next(self, event):
                self.ind += 1
                i = self.ind % Niter
                ax.cla()
                psi = bc.bc_psi_const(psi_hist[:, :, i], psi_0)
                ax.contourf(X,Y,psi,20,cmap=plt.get_cmap('coolwarm'))
                ax.contour(X,Y,psi,20,colors='k',alpha=0.2)
                ax.set_title('t = %0.3f' % t[i])
                plt.draw()

            def prev(self, event):
                self.ind -= 1
                i = self.ind % Niter
                ax.cla()
                psi = bc.bc_psi_const(psi_hist[:, :, i], psi_0)

                ax.contourf(X,Y,psi,20,cmap=plt.get_cmap('coolwarm'))
                ax.contour(X,Y,psi,20,colors='k',alpha=0.2)
                ax.set_title('t = %0.3f' % t[i])
                plt.draw()

        callback = Index()
        axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

        plt.show()