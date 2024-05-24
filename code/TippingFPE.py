import numpy as np
import pde
from itertools import product
from datetime import datetime


class TippingPDE(pde.PDEBase):

    def __init__(self, grid, coupling_matrix=None, noise_level=0.1):
        
        self.grid = grid                            # the simulation grid
        self.n = grid.dim                           # number of tipping systems
        self.coupling_matrix = coupling_matrix      # coupling matrix
        self.noise_level = noise_level              # noise level
        self.D = self.noise_level**2/2              # diffusion constant

        # generate component fields
        self.components = [pde.ScalarField(self.grid, self.grid.coordinate_arrays[i], label=str(i)) for i in range(self.n)]

        # potential force field
        self.potential_force = pde.VectorField.from_scalars([-v**3 + v for v in self.components])

        # coupling force field
        self.coupling_force = pde.VectorField.from_scalars([sum([coupling_matrix[i, j]*(self.components[j] + 1) for j in range(self.n) if i != j]) for i in range(self.n)])

        # sum force field
        self.force = self.potential_force + self.coupling_force

        # masks for the different states
        self.mask = []
        for mus in product([-1, 1], repeat=self.n):
            # generate mask for the quadrant
            mask = pde.ScalarField(self.grid, 1)
            for mu, v in zip(mus, self.components):
                mask *= pde.ScalarField(self.grid, np.sign(mu) == np.sign(v.data))
            self.mask.append(mask)

    def ic_all_states(self, sigma=0.1):
        ic = pde.ScalarField(self.grid, 0)
        for p in product([-1, 1], repeat=self.n):
            vfield = pde.ScalarField(self.grid, 1)
            for mu, v in zip(p, self.components):
                vfield *= pde.ScalarField(self.grid, np.exp(-(v-mu)**2/(2*sigma**2)))
            ic += vfield
        ic /= ic.integral
        return ic
    
    def ic_uniform(self):
        return pde.ScalarField(self.grid, 1/self.grid.volume)
    
    def ic_centered(self, sigma=1):
        ic = pde.ScalarField(self.grid, 1)
        for v in self.components:
            ic *= pde.ScalarField(self.grid, np.exp(-v**2/(2*sigma**2)))
        ic /= ic.integral
        return ic

    def evolution_rate(self, state, t=0):
        return -(state * self.force).divergence("dirichlet") + self.D*state.laplace("dirichlet")
    
    def _make_pde_rhs_numba(self, state):

        # fix parameters 
        D     = self.D
        force = self.force.data

        # create numba operators
        laplace    = state.grid.make_operator("laplace",    bc="dirichlet")
        divergence = state.grid.make_operator("divergence", bc="dirichlet")

        @pde.tools.numba.jit(cache=False)
        def pde_rhs(state_data, t=0):
            return -divergence(state_data*force) + D*laplace(state_data)
        
        return pde_rhs
    
    def state_probability(self, state): 
        return [(mask*state/state.integral).integral for mask in self.mask]
    
    def state_locations(self, state):
    
        def expc(var, p):
            return (var*p/p.integral).integral

        return [[expc(v, mask*state) for v in self.components] for mask in self.mask]
    

class ProbabilitySteadyStateTracker(pde.trackers.DataTracker):

    def __init__(self, eq, interrupts=1, tol=1e-3):
        super().__init__(self.get_statistic, interval=interrupts)

        self.p_last = None
        self.dp     = None
        self.tol    = tol
        self.n      = int(abs(np.log10(tol)))
        self.mus    = list(product([-1, 1], repeat=eq.n))
        self.eq     = eq

    def handle(self, state, time):
        
        if self.dp and self.dp <= self.tol:
            # print("state probability converged")
            raise pde.base.FinishedSimulation("state probabilities converged")
        
        return super().handle(state, time)
    
    def get_statistic(self, state, time):
        
        # current real time
        now = datetime.now()

        # calculate probabilities in each quadrant of the coordinate space
        state_probabilities = self.eq.state_probability(state)

        # convergence of state
        p_now = np.abs(state_probabilities).max()
        if self.p_last:
            self.dp = abs(self.p_last - p_now)
            # print(f"{now:%H:%M:%S}: p_max={p_now:>0.{self.n}f} |dp|={self.dp:>0.{self.n}f}")
        self.p_last = p_now

        #return {mu: p for mu, p in zip(self.mus, state_probabilities)}
        return [time, *state_probabilities]


def run_fpe_analysis(
                    coupling_matrix,
                    noise_level=0.1,
                    tmax=50, 
                    dt_initial=1e-4, 
                    adaptive=True, 
                    max_runtime_seconds=120, 
                    convergence_tol=1e-2,
                    p0="centered",
                    p0_sigma=0.2,
                    grid_extent=[[-3, 3], [-3, 3]],
                    grid_shape=[120, 120],
                    steady_state_interrupt=1):

        # initialize the grid
        grid  = pde.CartesianGrid(grid_extent, shape=grid_shape)

        # prepare the equation
        eq = TippingPDE(grid, coupling_matrix, noise_level)
        # initial condition
        if p0 == "centered":
            p0 = eq.ic_centered(p0_sigma)
        if p0 == "uniform":
            p0 = eq.ic_uniform()
        if p0 == "all_states" or p0 == "all":
            p0 = eq.ic_all_states(p0_sigma)

        # track convergence of the state probabilities
        statistics = ProbabilitySteadyStateTracker(eq, tol=convergence_tol)
        # limit total runtime
        runtime    = pde.trackers.RuntimeTracker(max_runtime_seconds, interval=steady_state_interrupt)

        # integrate coupling setup
        p, info = eq.solve(p0, t_range=tmax, dt=dt_initial, adaptive=adaptive, tracker=(statistics, runtime), ret_info=True)
        
        # summarize results of the simulation
        summary = (p.data, info, statistics.data, eq.state_locations(p))

        del grid, eq, statistics, p, info

        return summary