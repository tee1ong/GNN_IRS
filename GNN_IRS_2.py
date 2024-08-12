import numpy as np

# Number of BS antennas, IRS elements, and users
n_bs = 8  # Number of BS antennas
n_irs = 100  # Number of IRS elements
n_ue = 3  # Number of users
n_uet = 1 # number of antennas per user

# Power, signal config (in dBm)
Pt_up = 15 # Uplink transmit power
Pt_down = 20 # Downlink transmit power
Po_up = -100 # Uplink noise power
Po_down = -65 # Downlink Noise Power 
epsilon = 10 # Rician factor 
lambda_c = 1; # Relative wavelength

# Object Locations
bs_loc = np.array([100, -100, 0])  # BS location
irs_loc = np.array([0, 0, 0])  # IRS center location

# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements - presumed

# Function definitions 

# User locations
def generate_user_locations(n_ue):
    x = np.random.uniform(5, 35, n_ue) # x coordinate for users randomly distributed in a plane 
    y = np.random.uniform(-35, 35, n_ue) # y coordinate for users randomly distributed in a plane
    z = np.full(n_ue, -20) # constant z coordinate for the users
    return np.vstack((x, y, z)).T

# IRS Locations
def generate_irs_positions(irs_loc, n_rows, n_cols, d_irs):
    y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs) # RIS elements y coordinates (horizontal)
    z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs) # RIS elements z coordinates (vertical)
    y, z = np.meshgrid(y_coords, z_coords) # y_coords x z_coords
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, irs_loc[0]) # x plane at 0 
    return np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T # combine into an array of IRS element positions

# Direct channel path loss model 
def path_loss_direct(d):
    pl = 32.6 + 36.7 * np.log10(d)
    return pl

# Cascaded channel path loss model
def path_loss_cascaded(d):
    pl = 30 + 22 * np.log10(d)
    return pl

# Euclidean distance calculation 
def calculate_distance(a,b): 
    dist = np.linalg.norm(b - a)
    return dist

def calculate_angles(ue_loc, irs_loc, d_iu):
    phi_3_k = np.arctan2(ue_loc[:, 1] - irs_loc[1], ue_loc[:, 0] - irs_loc[0])  # Azimuth angle for IRS
    theta_3_k = np.arctan2(ue_loc[:, 2] - irs_loc[2], d_iu)  # Elevation angle for IRS
    return phi_3_k, theta_3_k

def steering_vector_irs(phi_3_k, theta_3_k, m, n, d_irs, lambda_c):
    i1 = (n % 10)  # Horizontal index (row) in the IRS panel
    i2 = (n // 10)  # Vertical index (column) in the IRS panel
    
    exponent = 2j * np.pi * d_irs / lambda_c * (i1 * np.sin(phi_3_k) * np.cos(theta_3_k) + i2 * np.sin(theta_3_k))
    return np.exp(exponent)

# Generates direct channel coefficients
def direct_channel(n_ue, n_uet, n_bs, beta_0):
    
    if len(beta_0) != n_ue:
        raise ValueError(f"Incompatible transmit power vectors")
    
    h = np.zeros((n_ue, n_uet, n_bs), dtype = complex)
    
    for k in range(n_ue):
        r_part = np.random.randn(n_uet, n_bs)
        i_part = np.random.randn(n_uet, n_bs)
        h[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * beta_0[k]
        
    return h

def IRS_UE_channel(n_ue, n_uet, n_irs, beta_1, epsilon, phi_3_k, theta_3_k, d_irs, lambda_c):
    
    if len(beta_1) != n_ue:
        raise ValueError("Incompatible beta_1 vector size with the number of users")
    
    # Compute the Rician factors
    rician_factor_los = np.sqrt(epsilon / (1 + epsilon))
    rician_factor_nlos = np.sqrt(1 / (1 + epsilon))
    
    # Initialize the channel matrices
    hkr_nlos = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    hkr_los = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    
    # Generate the NLOS component
    for k in range(n_ue):
        r_part = np.random.randn(n_uet, n_irs)
        i_part = np.random.randn(n_uet, n_irs)
        hkr_nlos[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * rician_factor_nlos
    
    # Generate the LOS component using the steering vectors
    for k in range(n_ue):  # For each user
        for m in range(n_uet):  # For each user antenna
            for n in range(n_irs):  # For each IRS element
                hkr_los[k, m, n] = steering_vector_irs(phi_3_k[k], theta_3_k[k], m, n, d_irs, lambda_c) * rician_factor_los
    
    # Combine the LOS and NLOS components, and apply path loss beta_1
    hkr = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    for k in range(n_ue):
        hkr[k, :, :] = beta_1[k] * (hkr_los[k, :, :] + hkr_nlos[k, :, :])
    
    return hkr
        
    
# Simulation

ue_loc = generate_user_locations(n_ue) # Generate user locations 
irs_positions = generate_irs_positions(irs_loc, n_rows, n_cols, d_irs)  # Generate 

# Ok, now get distances 
# Direct channel distance BS-UE
d_bu = np.zeros(n_ue) # Direct channel distance storage 

for k in range(n_ue):     
    d_bu[k] = calculate_distance(bs_loc, ue_loc[k])  # Direct channel distance calculation 
    
d_bi = calculate_distance(bs_loc,irs_loc) # Distance BS-IRS 

d_iu = np.zeros(n_ue) # Distance IRS-UE storage

for k in range(n_ue):
    d_iu[k] = calculate_distance(irs_loc,ue_loc[k]) # Distance IRS-UE

# Path losses
pl_bu = path_loss_direct(d_bu) # direct channel path loss 
pl_bi = path_loss_cascaded(d_bi) # BS-IRS channel path loss
pl_iu = path_loss_cascaded(d_iu) # IRS-UE channel path loss 
pl_cascaded = pl_bi + pl_iu # total cascaded channel path loss

pl_bu_linear = 10 ** (-pl_bu / 10)  # Linear scale direct channel path loss 

# Channel coefficients calculation
# Direct Channel
hkd = direct_channel(n_ue, n_uet, n_bs, pl_bu_linear)

# Cascaded channel
phi_3_k, theta_3_k = calculate_angles(ue_loc, irs_loc, d_iu) # azimuth and elevation angles between IRS and UE 

# a_irs = np.zeros((n_ue, n_uet, n_irs,), dtype=complex) # Steering angles = nlos channel coefficients storage array

# for k in range(n_ue):  # For each user
#     for m in range(n_uet):  # For each user antenna
#         for n in range(n_irs):  # For each IRS element
#             a_irs[k, m, n] = steering_vector_irs(phi_3_k[k], theta_3_k[k], m, n, d_irs, lambda_c)
hkr = IRS_UE_channel(n_ue, n_uet, n_irs, pl_iu, epsilon, phi_3_k, theta_3_k, d_irs, lambda_c)

        