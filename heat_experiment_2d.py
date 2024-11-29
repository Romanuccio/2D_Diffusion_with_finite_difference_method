import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags_array
from scipy.sparse.linalg import factorized


class CoefficientMask():
    # for now it's a mask for uniform grids
    def __init__(self, mask, value):
        self.mask = mask
        self.value = value


class HeatExperiment2D():
    def __init__(self, x_nodes=100, y_nodes=100, length_x=1, length_y=1,
                 terminal_time=3, timestep=0.01, x_thermal_diffusivity=23e-6, y_thermal_diffusivity=23e-6,
                 boundary_value = 273, boundary_conditions=False, initial_value=273, vein_masks=None, vein_function_values=None,
                 cmap='hot'):
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
        self.boundary_conditions = boundary_conditions
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
        
        # if there are no provided mask, use identity matrix
        # TODO could be optimised in calculations? tiny impact probably
        if vein_masks is None:
            self.vein_masks = np.array(np.ones([x_nodes, y_nodes]).flatten())
            self.vein_function_values = np.ones_like(self.vein_masks)
        else:
            self.vein_masks = np.array([mask.flatten() for mask in vein_masks])
            self.vein_function_values = vein_function_values
            
        self.initial_values = u
        self.apply_vein_masks(self.initial_values, 0)
        
        # BC
        if self.boundary_conditions:
            # top bottom
            self.solution[0][0:self.nx] = self.boundary_value
            self.solution[0][-1:-1-self.nx:-1] = self.boundary_value
            # # left right
            self.solution[0][::self.nx] = self.boundary_value
            self.solution[0][self.nx-1::self.nx] = self.boundary_value
        
    def run_calculation(self):
        SLHS, SRHS = self.setup_scheme()
        flat_u = self.initial_values
        self.solution = [self.initial_values]
        t = 0
        iteration = 0
        solve = factorized(SLHS)
        
        while t < self.tmax:
            b = SRHS@flat_u
            sol = solve(b)
            t += self.time_step
            
            # after calculating condition, set:
            
            # boundary conditions
            if self.boundary_conditions:
                # top bottom
                sol[0:self.nx] = self.boundary_value
                sol[-1:-1-self.nx:-1] = self.boundary_value
                # # left right
                sol[::self.nx] = self.boundary_value
                sol[self.nx-1::self.nx] = self.boundary_value
            
            # masks
            # TODO fix this retarded logic
            iteration += 1
            if iteration < len(self.vein_function_values):
                self.apply_vein_masks(sol, iteration)
            
            self.solution.append(sol)
            flat_u = sol
        
        self.solution = [solution.reshape(self.nx, self.ny) for solution in self.solution]

    def apply_vein_masks(self, sol, iteration):
        # sets the value of the solution to the value of the mask
        #TODO ravel over flatten!
        for mask in self.vein_masks:
            np.copyto(sol, self.vein_function_values[iteration]*mask, where=mask != 0)

    def setup_scheme(self):
        alpha = self.x_thermal_diffusivity # m^2/s
        beta = self.y_thermal_diffusivity # m^2/s
        A = self.time_step*alpha/(self.dx**2)
        B = self.time_step*beta/(self.dy**2)
        C = self.time_step*(alpha*(self.dy**2) + beta*(self.dx**2))/((self.dx**2) * (self.dy**2))
        C_LHS = 2*(1+C)
        C_RHS = 2*(1-C)

        # matrices setup
        
        #TODO stop using central differences on edges...??
        main_diag = np.full(self.dim, C_LHS)
        x_off_diag = np.full(self.dim - 1, -A)
        x_off_diag[self.nx-1::self.nx] = 0
        y_off_diag = np.full(self.dim - self.nx, -B)

        diagonals = [main_diag, x_off_diag, x_off_diag, y_off_diag, y_off_diag]
        offsets = [0, 1, -1, self.nx, -self.nx]
        SLHS = diags_array(diagonals, offsets=offsets).tocsc()
        main_diag = np.full(self.dim, C_RHS)
        diagonals = [main_diag, -x_off_diag, -x_off_diag, -y_off_diag, -y_off_diag]
        SRHS = diags_array(diagonals, offsets=offsets).tocsc()
        
        return SLHS, SRHS

    def visualize_animated(self, save=False):

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
            vmax=np.max(self.solution),
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
        if save:
            self.anim.save('vein.gif', writer='pillow')

        # Show animation
        plt.show()
        
    def visualize_snapshots(self, timestamps = None):
        if timestamps is None:
            timestamps = [0, round(len(self.solution)/3), 2*round(len(self.solution)/3), -1]
            
        fig, axpairs = plt.subplots(2, 2)
        for pair in axpairs:
            for ax in pair:
                ax.set_aspect('equal')
                # ax.set_xticks(np.linspace(0, 1, 10)*self.lx)
                # ax.set_xlim(0, self.lx)
                # ax.set_yticks(np.linspace(0, 1, 10)*self.ly)
                # ax.set_ylim(0, self.ly)
                im = ax.imshow(self.solution[timestamps.pop(0)], vmin=np.min(self.solution), vmax=np.max(self.solution), cmap=self.cmap)
                fig.colorbar(im)
        plt.show()