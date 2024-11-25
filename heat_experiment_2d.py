import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

class HeatExperiment2D():
    def __init__(self, x_nodes=100, y_nodes=100, length_x=1, length_y=1,
                 terminal_time=3, timestep=0.01, x_thermal_diffusivity=23e-6, y_thermal_diffusivity=23e-6,
                 boundary_value = 273, initial_value=273, masks=None, cmap='hot'):
        u = initial_value*np.ones([x_nodes, y_nodes])
        # for i in range (-10, 10):
        #     for j in range(-10, 10):
        #         u[50+i,50+j] += 600
                
        # for i in range (-5, 5):
        #     for j in range(-5, 5):
        #         u[55+i,55+j] += 1200
        
        # for i in range (-5, 5):
        #     for j in range(-5, 5):
        #         u[65+i,65+j] += 400
                
        u = u.flatten()
        
        self.cmap = plt.get_cmap(cmap)
        self.nx = x_nodes
        self.ny = y_nodes
        self.dim = self.nx*self.ny
        self.lx = length_x
        self.ly = length_y
        self.dx = length_x/x_nodes
        self.dy = length_y/y_nodes
        self.tmax = terminal_time
        self.x_thermal_diffusivity = x_thermal_diffusivity
        self.y_thermal_diffusivity = y_thermal_diffusivity
        self.time_step = timestep
        self.boundary_value = boundary_value
        self.solution = []
        if masks is None:
            self.masks = np.array(np.ones([x_nodes, y_nodes]).flatten())
        else:
            self.masks = np.array([mask.flatten() for mask in masks])
            
        self.initial_values = u
        self.apply_masks(self.initial_values)
        
    def run_calculation(self):
        SLHS, SRHS = self.setup_scheme()
        flat_u = self.initial_values
        self.solution = [self.initial_values]
        t = self.tmax
        solve = factorized(SLHS)
        while t > 0:
            b = SRHS@flat_u
            # sol = sp.sparse.linalg.spsolve(SLHS, b)
            sol = solve(b)
            t -= self.time_step
            
            # boundary conditions #TODO move to init
            # top bottom
            sol[0:self.nx] = self.boundary_value
            sol[-1:-1-self.nx:-1] = self.boundary_value
            # left right
            sol[::self.nx] = self.boundary_value
            sol[self.nx-1::self.nx] = self.boundary_value
            
            # masks
            self.apply_masks(sol)
            
            self.solution.append(sol)
            flat_u = sol
        
        self.solution = [solution.reshape(self.nx, self.ny) for solution in self.solution]

    def apply_masks(self, sol):
        for mask in self.masks:
            np.copyto(sol, mask, where=mask != 0)

    def setup_scheme(self):
        alpha = self.x_thermal_diffusivity # m^2/s
        beta = self.y_thermal_diffusivity # m^2/s
        A = self.time_step*alpha/(self.dx**2)
        B = self.time_step*beta/(self.dy**2)
        C = self.time_step*(alpha*(self.dy**2) + beta*(self.dx**2))/((self.dx**2) * (self.dy**2))
        C_LHS = 2*(1+C)
        C_RHS = 2*(1-C)

        # matrices setup
        
        main_diag = np.full(self.dim, C_LHS)
        x_off_diag = np.full(self.dim - 1, -A)
        y_off_diag = np.full(self.dim - self.nx, -B)

        diagonals = [main_diag, x_off_diag, x_off_diag, y_off_diag, y_off_diag]
        offsets = [0, -1, 1, -self.nx, self.nx]
        SLHS = diags(diagonals, offsets, shape=(self.dim, self.dim)).tocsc()
        main_diag = np.full(self.dim, C_RHS)
        diagonals = [C_RHS, -x_off_diag, -x_off_diag, -y_off_diag, -y_off_diag]
        SRHS = diags(diagonals, offsets, shape=(self.dim, self.dim)).tocsc()
        # LHS = np.zeros((self.dim,self.dim))
        # np.fill_diagonal(LHS, C_LHS)
        # first_diag = np.arange(self.dim-1)
        # second_diag = np.arange(self.nx, self.dim)
        # LHS[first_diag + 1, first_diag] = -A
        # LHS[second_diag - self.nx, second_diag] = -B
        # LHS = (LHS + LHS.transpose()) - np.diag(LHS.diagonal())
        # RHS = LHS.copy()
        # RHS *= -1
        # np.fill_diagonal(RHS, C_RHS)
        # # SLHS = sp.sparse.csr_array(LHS)
        # SRHS = sp.sparse.csr_array(RHS)
        
        return SLHS, SRHS

    def visualize_animated(self):

        # Define the target number of frames for the animation
        target_frames = 100

        # Determine the indices of the frames to pick from the solution
        num_frames = len(self.solution)
        selected_indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)

        # Calculate interval dynamically based on the total duration and the number of target frames
        interval = 33  # Interval in milliseconds

        # Create a figure and axis
        fig, ax = plt.subplots()
        heatmap = ax.imshow(
            self.solution[0],
            vmin=np.min(self.solution),
            vmax=np.max(self.solution) + 50,
            cmap=self.cmap,
        )
        plt.colorbar(heatmap)

        # Update function for animation
        def update(frame):
            heatmap.set_array(self.solution[selected_indices[frame]])
            return [heatmap]

        # Create animation
        self.anim = FuncAnimation(
            fig, update, frames=range(target_frames), interval=interval, repeat=True, blit=True
        )

        # Show animation
        plt.show()

  # Display animation


    # def visualize_animated(self, frames=None, interval=1):
    #     if frames is None:
    #         frames = len(self.solution)
            
    #     fig, ax = plt.subplots(layout='constrained')
    #     x = np.arange(self.nx)
    #     y = np.arange(self.ny)
    #     pcm = ax.pcolormesh(x, y, self.solution[0], shading='gouraud', vmin=np.min(self.solution), vmax=np.max(self.solution)+50, cmap='hot')
    #     fig.colorbar(pcm)
    #     ax.set_aspect('equal')

    #     def update(frame):
    #         pcm.set_array(self.solution[frame].flatten())
    #         return [pcm]

    #     self.anim = FuncAnimation(fig=fig, func=update, frames=len(self.solution), interval=interval, repeat=True,blit=False)
            
    #     plt.show()
        
    def visualize_snapshots(self, timestamps = None):
        if timestamps is None:
            timestamps = [0, round(len(self.solution)/3), 2*round(len(self.solution)/3), -1]
            
        fig, axpairs = plt.subplots(2, 2)
        for pair in axpairs:
            for ax in pair:
                ax.set_aspect('equal')
                im = ax.imshow(self.solution[timestamps.pop(0)], vmin=np.min(self.solution), vmax=np.max(self.solution)+50, cmap=self.cmap)
                fig.colorbar(im)
        plt.show()