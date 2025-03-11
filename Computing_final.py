# In this section I am importing all the libraries I will need
import numpy as np # For carrying out array and numerical computations
import matplotlib.pyplot as plt # For plotting static visualisation
import matplotlib.animation as anim  # For plotting animated visualisation
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
import scipy.sparse as sp
import scipy.sparse.linalg as spla # Both sp, spla used for sparse matrix operations and solving linear systems in the implicit method.

# In this section I am setting the domain of solution and the discretised grid

L = 1.0          # Length of the square domain (both x and y directions)
T = 3.0          # Total simulation time
c = 1.0          # Wave speed (constant)
# In this example case, the wave equation is solved on a 1m * 1m square grid over 3 seconds

# Discretising the domain
nx, ny = 51, 51          # Number of grid points in x and y directions (Domain divided into 51*51 points)
dx = L / (nx - 1)        # Grid spacing in x-direction 
dy = dx                  # Grid spacing in y-direction (Assumed equal to dx)

# Time stepping 
dt_exp = 0.005          # Time step for explicit finite difference method (have to satisfy CFL condition)
dt_impl = 0.01          # Time step for implicit method
CFL = c * dt_exp / dx   # Courant number (stability condition)
nt_exp = int(T / dt_exp) + 1  # Total explicit time steps
nt_impl = int(T / dt_impl) + 1  # Total implicit time steps

# Create spatial grid
x = np.linspace(0, L, nx) 
y = np.linspace(0, L, ny) # Gives equally spaced grid points
X, Y = np.meshgrid(x, y, indexing='ij') # Creates 2D coodrinate matrices 

# In this section I am defining arrays I would need (if needed)
# (Arrays for the numerical solution will be allocated below in each method)

# In this section I am setting the boundary conditions/initial values

# Define the initial displacement as a Gaussian pulse (drumhead)
u0 = 5 * np.exp(-100 * ((X - 0.5)**2 + (Y - 0.5)**2)) # The pulse is centred at (0.5, 0.5)
# Apply Dirichlet boundary conditions (fixed edges: u = 0)
u0[0, :], u0[-1, :], u0[:, 0], u0[:, -1] = 0, 0, 0, 0

###########################################################

# In this section I am implementing the numerical method

# Explicit Finite Difference Method
u_exp = np.zeros((nt_exp, nx, ny))  # Create a 3D array to store the wave at each time step
u_exp[0] = u0.copy()  # Set initial condition at t = 0 (first time step)

# First time step calculation 
laplacian_u0_exp = np.zeros_like(u0)
# Use five-point stencil to compute the Laplacian (second derivative)
laplacian_u0_exp[1:-1, 1:-1] = (
    u_exp[0, 2:, 1:-1] + u_exp[0, :-2, 1:-1] +
    u_exp[0, 1:-1, 2:] + u_exp[0, 1:-1, :-2] -
    4 * u_exp[0, 1:-1, 1:-1]
) / (dx**2)
# Estimate the second time step with taylor series expansion
u_exp[1, 1:-1, 1:-1] = u0[1:-1, 1:-1] + 0.5 * (c**2 * dt_exp**2) * laplacian_u0_exp[1:-1, 1:-1] 

# Time-stepping loop for the explicit method 
# Iterates over time step to update the 3D array using explicit finite difference scheme
for n in range(1, nt_exp - 1):
    laplacian_un = np.zeros_like(u_exp[n])
    laplacian_un[1:-1, 1:-1] = (
        u_exp[n, 2:, 1:-1] + u_exp[n, :-2, 1:-1] +
        u_exp[n, 1:-1, 2:] + u_exp[n, 1:-1, :-2] -
        4 * u_exp[n, 1:-1, 1:-1]
    ) / (dx**2)
    u_exp[n+1, 1:-1, 1:-1] = (
        2 * u_exp[n, 1:-1, 1:-1] - u_exp[n-1, 1:-1, 1:-1] +
        (c**2 * dt_exp**2) * laplacian_un[1:-1, 1:-1]
    )


# Implicit Finite Difference Method

# Set up a sparse system matrix using kronecker products
alpha_impl = c**2 * dt_impl**2 / dx**2  # Coefficient in the implicit scheme

u_impl = np.zeros((nt_impl, nx, ny))  # Create solution array for implicit method
u_impl[0] = u0.copy()

# Construct the system matrix for the interior nodes (implicit finite difference method)
nx_i = nx - 2  # Interior points in x
ny_i = ny - 2  # Interior points in y

main_diag = (1 + 4 * alpha_impl) * np.ones(nx_i)
off_diag = -alpha_impl * np.ones(nx_i - 1)
Tx = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')
Ty = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')

I_nx = sp.eye(nx_i, format='csc')
I_ny = sp.eye(ny_i, format='csc')
A_matrix = sp.kron(I_ny, Tx) + sp.kron(Ty, I_nx)

# Time-stepping loop for the implicit method
for n in range(1, nt_impl - 1):
    b = 2 * u_impl[n, 1:-1, 1:-1] - u_impl[n-1, 1:-1, 1:-1]
    b = b.flatten()  # Flatten the interior grid into a vector
    u_int_next = spla.spsolve(A_matrix, b)
    u_impl[n+1, 1:-1, 1:-1] = u_int_next.reshape((nx_i, ny_i))


#########################################################################

# In this section I am showing the results

# Define parameters
time_points = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0]
figsize = (3 * len(time_points), 6)

# Static visualisation for Explicit and Implicit Methods
def static_plot(method, u, dt):
    fig, axs = plt.subplots(2, len(time_points), figsize=figsize)
    for i, t in enumerate(time_points):
        idx = int(t / dt)
        cp = axs[0, i].contourf(X, Y, u[idx, :, :], levels=50, cmap='viridis')
        axs[0, i].set_title(f'{method}: t = {t:.2f}s')
        axs[0, i].set_xlabel('x')
        axs[0, i].set_ylabel('y')
        axs[0, i].set_aspect('equal', 'box')
        axs[1, i].plot(x, u[idx, :, ny // 2], color='b')
        axs[1, i].set_title(f'Cross-section (y=0.5) at t={t:.2f}s')
        axs[1, i].set_xlabel('x')
        axs[1, i].set_ylabel('Displacement')
    
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([1.01, 0.55, 0.02, 0.3])
    fig.colorbar(cp, cax=cbar_ax).set_label('Displacement', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

# Plot both graph
static_plot("Explicit", u_exp, dt_exp)
static_plot("Implicit", u_impl, dt_impl)

# Animated 3D surface plot function
def animated_plot(method, u, dt, filename, interval, frame_per_sec):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Displacement')
    ax.set_zlim(-6, 6)

    def update(frame):
        ax.clear()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Displacement')
        ax.set_zlim(-6, 6)
        ax.plot_surface(X, Y, u[frame, :, :], cmap='viridis', rstride=1, cstride=1)
        ax.set_title(f"{method} Method: t = {frame * dt:.2f} s")
    
    ani = anim.FuncAnimation(fig, update, frames=range(0, len(u), 5), interval=interval)
    ani.save(filename, writer='pillow', fps= frame_per_sec)
    plt.show()

# Plot for both graphs
animated_plot("Explicit", u_exp, dt_exp, "explicit_3d_animation.gif", interval=25, frame_per_sec=15)
animated_plot("Implicit", u_impl, dt_impl, "implicit_3d_animation.gif", interval=200, frame_per_sec=5)

# 3D Surface Plot at t=0 for Implicit Method
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u_impl[0, :, :], cmap='viridis', rstride=1, cstride=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Displacement')
ax.set_title("Implicit Method: t = 0.00 s")
ax.set_zlim(-6, 6)
fig.colorbar(ax.plot_surface(X, Y, u_impl[0, :, :], cmap='viridis', rstride=1, cstride=1), shrink=0.5, aspect=10).set_label('Displacement')
plt.show()


# In this section I am celebrating
print('CW done: I deserve a good mark')