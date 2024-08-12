import numpy as np

# Number of BS antennas, IRS elements, and users
n_bs = 8  # Number of BS antennas
n_irs = 100  # Number of IRS elements
n_ue = 3  # Number of users

# Object Locations
bs_loc = np.array([100, -100, 0])  # BS location
irs_loc = np.array([0, 0, 0])  # IRS center location

# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements

# User locations
def generate_user_locations(n_ue):
    x = np.random.uniform(5, 35, n_ue)
    y = np.random.uniform(-35, 35, n_ue)
    z = np.full(n_ue, -20)
    return np.vstack((x, y, z)).T

def generate_irs_positions(irs_loc, n_rows, n_cols, d_irs):
    y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs)
    z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs)
    y, z = np.meshgrid(y_coords, z_coords)
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, irs_loc[0])
    return np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T

ue_loc = generate_user_locations(n_ue)
irs_positions = generate_irs_positions(irs_loc, n_rows, n_cols, d_irs)

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Path-loss model
def path_loss(distance):
    return 10 ** (-3.0) * (distance ** -3.5)

# Function to generate channels considering IRS element positions
def generate_channel(ue_loc, bs_loc, irs_positions, n_bs, n_irs, n_ue):
    h_d = np.zeros((n_ue, n_bs), dtype=complex)
    h_r = np.zeros((n_ue, n_irs), dtype=complex)
    G = np.zeros((n_bs, n_irs), dtype=complex)
    
    for k in range(n_ue):
        # Direct BS to UE channel
        d_bs_ue = calculate_distance(bs_loc, ue_loc[k])
        path_loss_bs_ue = path_loss(d_bs_ue)
        h_d[k, :] = (np.random.randn(n_bs) + 1j * np.random.randn(n_bs)) * np.sqrt(path_loss_bs_ue / 2)
        
        # IRS to UE channel
        for i in range(n_irs):
            d_irs_ue = calculate_distance(irs_positions[i], ue_loc[k])
            path_loss_irs_ue = path_loss(d_irs_ue)
            h_r[k, i] = (np.random.randn() + 1j * np.random.randn()) * np.sqrt(path_loss_irs_ue / 2)
        
        # BS to IRS channel (this is the same for all users)
        for i in range(n_irs):
            d_bs_irs = calculate_distance(bs_loc, irs_positions[i])
            path_loss_bs_irs = path_loss(d_bs_irs)
            G[:, i] = (np.random.randn(n_bs) + 1j * np.random.randn(n_bs)) * np.sqrt(path_loss_bs_irs / 2)
    
    return h_d, h_r, G

# Generate channels
hd, hr, G = generate_channel(ue_loc, bs_loc, irs_positions, n_bs, n_irs, n_ue)

print("Direct BS to UE channel (hd):\n", hd)
print("IRS to UE channel (hr):\n", hr)
print("BS to IRS channel (G):\n", G)
