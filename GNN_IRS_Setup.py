import numpy as np

n_bs = 8  # number of BS antennas
n_irs = 100  # no. of IRS elements
n_ue = 3  # no. of users

# Object Locations 
bs_loc = np.array([100, -100, 0])  # BS location 
irs_loc = np.array([0, 0, 0])  # IRS center location 

# User locations 
ue_x = np.random.uniform(5, 35, n_ue)
ue_y = np.random.uniform(-35, 35, n_ue)
ue_z = np.full(n_ue, -20)

ue_loc = np.vstack((ue_x, ue_y, ue_z)).T

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

# Direct Channel path losses
d_bu = np.zeros(n_ue)

for k in range(n_ue):
    d_bu_k = np.linalg.norm(ue_loc[k] - bs_loc)
    d_bu[k] = d_bu_k

pl_bu = 32 + 36.7 * np.log10(d_bu)

# IRS-assisted channel path losses 
d_bi = np.linalg.norm(irs_loc - bs_loc)

d_iu  = np.zeros(n_ue)

for k in range(n_ue):
    d_iu_k = np.linalg.norm(ue_loc[k] - irs_loc)
    d_iu[k] = d_iu_k

pl_bi = 30 + 22 * np.log10(d_bi)
pl_iu = 30 + 22 * np.log10(d_iu)

# Steering Vectors for IRS
lambda_c = 1  # normalized wavelength
phi_3 = np.arctan2(ue_y - irs_loc[1], ue_x - irs_loc[0])  # azimuth angles for users
theta_3 = np.arctan2(ue_z - irs_loc[2], np.sqrt((ue_x - irs_loc[0])**2 + (ue_y - irs_loc[1])**2))  # elevation angles

# Initialize IRS steering vectors
a_irs = np.zeros((n_ue, n_irs), dtype=complex)

for k in range(n_ue):
    for n in range(n_irs):
        i1 = n % 10  # index along y direction
        i2 = n // 10  # index along z direction
        a_irs[k, n] = np.exp(1j * 2 * np.pi * d_irs * (i1 * np.sin(phi_3[k]) * np.cos(theta_3[k]) + i2 * np.sin(theta_3[k])) / lambda_c)

# Steering Vector for BS
phi_1 = np.arctan2(bs_loc[1] - irs_loc[1], bs_loc[0] - irs_loc[0])
theta_1 = np.arctan2(bs_loc[2] - irs_loc[2], np.sqrt((bs_loc[0] - irs_loc[0])**2 + (bs_loc[1] - irs_loc[1])**2))
a_bs = np.exp(1j * 2 * np.pi * d_irs * np.arange(n_bs) * np.cos(phi_1) * np.cos(theta_1) / lambda_c)

# Output the steering vectors (for verification)
print("Steering vectors for IRS (for each user):", a_irs)
print("Steering vector for BS:", a_bs)

