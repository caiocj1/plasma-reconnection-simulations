import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.stats
import time
from datetime import datetime
import numpy as np
import calc
import bc

class Reconnection:

    # ----- Modelling and running simulations ----- #

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
        if nx % 2 != 0 or ny % 2 != 0:
            raise Exception('Even resolution required in both directions')

        k = 10.0
        x = np.linspace(-1, 1, nx + 2)
        y = np.linspace(-1, 1, ny + 2)

        Y, X = np.meshgrid(y,x)

        psi_0 = np.log(np.cosh(k*Y))/k
        j_0 = k/np.cosh(k*Y)**2
        
        return psi_0, j_0, 1e-3, 1e-4

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
        CPsij = calc.CrochetPoisson(self.psi_0, dj, self.dx, self.dy) + calc.CrochetPoisson(dpsi, self.j_0, self.dx, self.dy) + calc.CrochetPoisson(dpsi, dj, self.dx, self.dy)

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
        self.psi_hist = np.zeros((self.nx, self.ny, Niter))
        A = 1e-7
        dPsi = A*np.cos(2*np.pi*self.X)
        self.psi_hist[:, :, 0] = (self.psi_0 + dPsi)[1:-1, 1:-1]

        self.omega_hist = np.zeros((self.nx, self.ny, Niter))
 
        for i in range(Niter-1):
            cur = np.zeros((self.nx, self.ny, 2))
            cur[:, :, 0] = self.psi_hist[:, :, i]
            cur[:, :, 1] = self.omega_hist[:, :, i]

            new_x = self.RKTimeStep(cur, dt)
            self.t[i+1] = self.t[i]+dt
            self.psi_hist[:, :, i+1] = new_x[:, :, 0]
            self.omega_hist[:, :, i+1] = new_x[:, :, 1]

        print('run() done, %0.3f' % (time.time() - start_time), 's')

    # ----- Displaying results, post-treatment methods ----- #

    def dpsi_center(self) -> np.ndarray:
        """
        Evolution of psi in center of current sheet
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]
        dpsi = np.zeros(Niter)
        for i in range(Niter):
            dpsi[i] = self.psi_hist[self.nx//2, self.ny//2, i] - self.psi_0[self.nx//2, self.ny//2]
        return(dpsi)

    def plot_dpsi_center(self) -> None:
        """
        Plot evolution of psi in center of current sheet
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]
        dt = self.t[1] - self.t[0]
        dpsi = np.zeros(Niter)
        for i in range(Niter):
            dpsi[i] = self.psi_hist[self.nx//2, self.ny//2, i] - self.psi_0[self.nx//2, self.ny//2]
        plt.figure(1)
        plt.clf()
        ax = plt.axes()
        ax.cla()
        ax.set_xlabel('t')
        ax.set_ylabel('$\Delta \psi$')
        plt.semilogy(self.t, np.abs(dpsi), lw=2)
        plt.savefig(f'results/img/{self.nx}x{self.ny}_eta{self.eta}_nu{self.nu}_{Niter}_dt{dt}.png')
        plt.show()

    def linfit_dpsi_center(self, start: float, end: float) -> float:
        """
        Calculates linear slope between start and end points
        Prints slope and std error to text file
        Supposed to be used after plotting to see adequate inputs
        @return : slope of linear regression
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')

        if start < self.t[0] or end > self.t[len(self.t) - 1]:
            raise Exception('Start or end time out of bounds.')

        Niter = self.t.shape[0]
        dt = self.t[1] - self.t[0]

        start_idx = (int) (start / dt)
        end_idx = (int) (end / dt)

        dpsi = np.zeros(Niter)
        for i in range(Niter):
            dpsi[i] = self.psi_hist[self.nx//2, self.ny//2, i] - self.psi_0[self.nx//2, self.ny//2]

        slope, intercept, r, p, se = scipy.stats.linregress(self.t[start_idx:end_idx], np.log10(dpsi)[start_idx:end_idx])

        with open('results/results.txt', 'a') as f:
            print(f'{self.nx}x{self.ny}, eta = {self.eta}, nu = {self.nu}, {Niter} iterations, dt = {dt}', file=f)
            print('Slope log10:', slope, file=f)
            print('Std error:', se, file=f)
            print('\n', file=f)

        return slope
        
    def plot_sheet(self) -> None:
        """
        Plot contour levels of phi and psi across current sheet
        Has buttons for time control and to save specific instant
        """
        if not hasattr(self, 't'):
            raise Exception('Required attributes absent. Use run() first.')
        Niter = self.t.shape[0]

        omega_hist = self.omega_hist
        psi_hist, psi_0 = self.psi_hist, self.psi_0
        dx, dy, X, Y = self.dx, self.dy, self.X, self.Y
        t = self.t

        fig, ax = plt.subplots(1, 2)
        fig.subplots_adjust(bottom=0.2)
        fig.set_size_inches(13,6)

        def update(i):
            ax[0].cla()
            ax[1].cla()

            omega = bc.bc_omega_zero(omega_hist[:, :, i])
            phi = calc.LapInv(omega, dx, dy)
            psi = bc.bc_psi_const(psi_hist[:, :, i], psi_0)

            ax[0].contourf(X,Y,phi,20,cmap=plt.get_cmap('coolwarm'))
            ax[0].contour(X,Y,phi,20,colors='k',alpha=0.2)
            ax[0].set_title(r'$\phi$, t = %0.3f' % t[i])

            ax[1].contourf(X,Y,psi,20,cmap=plt.get_cmap('coolwarm'))
            ax[1].contour(X,Y,psi,20,colors='k',alpha=0.2)
            ax[1].set_title(r'$\psi$, t = %0.3f' % t[i])

        update(0)

        class Index():
            ind = 0

            def save(self, event):
                now = datetime.today().strftime('%d_%m_%Y-%H_%M_%S')
                plt.savefig(f'results/img/current_sheet_{now}.png')

            def zero(self, event):
                self.ind = 0
                update(0)
                plt.draw()
            
            def mid(self, event):
                self.ind = Niter//2
                update(self.ind)
                plt.draw()

            def next(self, event):
                self.ind += 1
                i = self.ind % Niter
                update(i)
                plt.draw()

            def prev(self, event):
                self.ind -= 1
                i = self.ind % Niter
                update(i)
                plt.draw()

        callback = Index()
        axprev = fig.add_axes([0.70, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        axmidp = fig.add_axes([0.09, 0.05, 0.1, 0.075])
        axzero = fig.add_axes([0.20, 0.05, 0.1, 0.075])
        axsave = fig.add_axes([0.31, 0.05, 0.1, 0.075])

        bzero = Button(axzero, 'Init')
        bzero.on_clicked(callback.zero)

        bmidp = Button(axmidp, 'Mid')
        bmidp.on_clicked(callback.mid)

        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)

        bprev = Button(axprev, 'Prev')
        bprev.on_clicked(callback.prev)

        bsave = Button(axsave, 'Save')
        bsave.on_clicked(callback.save)

        plt.show()