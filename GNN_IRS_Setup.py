import numpy as np

n_bs = 8 # number of BS antennas
n_irs = 100 # no. of IRS elements
n_ue = 3 # no. of users

# Object Locations 
bs_loc = np.array([100, -100, 0]) # BS location 
irs_loc = np.array([0, 0, 0]) # IRS center location 

# User locations 
ue_x = np.random.uniform(5, 35, n_ue)
ue_y = np.random.uniform(-35, 35, n_ue)
ue_z = np.full(n_ue, -20)

ue_loc = np.vstack((ue_x,ue_y,ue_z)).T

# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements (normalized to wavelength, typically λ/2)

y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs)
z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs)
y, z = np.meshgrid(y_coords, z_coords)
y = y.flatten()
z = z.flatten()
x = np.full_like(y, irs_loc[0])
irs_positions = np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T


