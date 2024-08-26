import numpy as np 
from scipy.linalg import dft
## Function Definitions 

# User and singular IRS elements locations 
# User locations
def generate_user_locations(n):
    x = np.random.uniform(5, 35, n) # x coordinates for users randomly distributed on x axis [5, 35]
    y = np.random.uniform(-35, 35, n) # y coordinates for users randomly distributed on y axis [-35, 35]
    z = np.full(n, -20) # z-coordinates for users capped at -20
    return np.vstack((x, y, z)).T

#  IRS elements Locations 
# IRS Locations
def generate_irs_positions(irs_loc, n_rows, n_cols, d_irs):
    y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs)  # RIS elements y coordinates (horizontal)
    z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs)  # RIS elements z coordinates (vertical)
    y, z = np.meshgrid(y_coords, z_coords)  # y_coords x z_coords
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, irs_loc[0])  # x plane at 0 
    return np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T  # combine into an array of IRS element positions

# Euclidean distance calculation 
def calculate_distance(a, b): 
    dist = np.linalg.norm(b - a)
    return dist

# Universal angle calculation function
def calculate_angles(point_a, point_b):
    d_ab = calculate_distance(point_a, point_b)
    phi = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    theta = np.arctan2(point_b[2] - point_a[2], d_ab)
    return phi, theta

# Universal steering vector function
def steering_vector(phi, theta, n, d_element, lambda_c, array_type='irs', n_cols=None):
    if array_type == 'irs':
        if n_cols is None:
            raise ValueError("n_cols must be provided for IRS steering vector calculation.")
        
        i1 = (n % n_cols)  # Horizontal index (row) in the IRS panel
        i2 = (n // n_cols)  # Vertical index (column) in the IRS panel
        exponent = 2j * np.pi * d_element / lambda_c * (i1 * np.sin(phi) * np.cos(theta) + i2 * np.sin(theta))
    
    elif array_type == 'bs':
        exponent = 2j * np.pi * n * d_element / lambda_c * np.cos(phi) * np.cos(theta)
    
    else:
        raise ValueError("Invalid array_type. Choose 'irs' or 'bs'.")
    
    return np.exp(exponent)

# Channel Coefficients
# Direct Channel 
def direct_channel(n_ue, n_uet, n_bs, beta_0):
    if len(beta_0) != n_ue:
        raise ValueError(f"Incompatible transmit power vectors")
    
    h = np.zeros((n_ue, n_uet, n_bs), dtype=complex) # storage for direct channel coefficients
    
    for k in range(n_ue):
        r_part = np.random.randn(n_uet, n_bs) 
        i_part = np.random.randn(n_uet, n_bs)
        h[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * beta_0[k] # ~CN(0,I)
        
    return h

# IRS-to-UE channel matrix calculation
def IRS_UE_channel(n_ue, n_uet, n_irs, beta_1, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols):
    if len(beta_1) != n_ue:
        raise ValueError("Incompatible beta_1 vector size with the number of users")
    
    rician_factor_los = np.sqrt(epsilon / (1 + epsilon))
    rician_factor_nlos = np.sqrt(1 / (1 + epsilon))
    
    hkr_nlos = np.zeros((n_ue, n_uet, n_irs), dtype=complex) # NLOS component 
    hkr_los = np.zeros((n_ue, n_uet, n_irs), dtype=complex) # LOS component 
    
    for k in range(n_ue):
        r_part = np.random.randn(n_uet, n_irs)
        i_part = np.random.randn(n_uet, n_irs)
        hkr_nlos[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * rician_factor_nlos
    
    for k in range(n_ue):  # For each user
        phi_3_k, theta_3_k = calculate_angles(irs_loc, ue_loc[k])
        for m in range(n_uet):  # For each user antenna
            for n in range(n_irs):  # For each IRS element
                hkr_los[k, m, n] = steering_vector(phi_3_k, theta_3_k, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols) * rician_factor_los
    
    hkr = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    for k in range(n_ue):
        hkr[k, :, :] = beta_1[k] * (hkr_los[k, :, :] + hkr_nlos[k, :, :])
    
    return hkr

# BS-IRS channel matrix calculation
def BS_IRS_channel(n_irs, n_bs, beta_2, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols):
    phi_1, theta_1 = calculate_angles(bs_loc, irs_loc)
    g_los = np.zeros((n_bs, n_irs), dtype=complex)
    for n in range(n_irs):
        a_bs = steering_vector(phi_1, theta_1, n, d_bs, lambda_c, array_type='bs')
        phi_2, theta_2 = calculate_angles(irs_loc, bs_loc)
        a_irs = steering_vector(phi_2, theta_2, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols)
        g_los[:, n] = np.outer(a_bs, a_irs.conj()).flatten()
    
    g_nlos = generate_nlos_component(n_bs, n_irs)
    g_full = beta_2 * (np.sqrt(epsilon / (1 + epsilon)) * g_los + np.sqrt(1 / (1 + epsilon)) * g_nlos)
    
    return g_full

# Function to generate NLOS component for G
def generate_nlos_component(n_bs, n_irs):
    return (np.random.randn(n_bs, n_irs) + 1j * np.random.randn(n_bs, n_irs)) / np.sqrt(2)

# Path loss models 

# Direct channel path loss model 
def path_loss_direct(d):
    pl = 32.6 + 36.7 * np.log10(d)
    return pl

# Cascaded channel path loss model
def path_loss_cascaded(d):
    pl = 30 + 22 * np.log10(d)
    return pl

## Uplink pilot transmission

# Generate orthogonal pilot sequences
def generate_phase_shifts(L, n_ue, n_ris):
    L0 = n_ue # number of symbols per subframe (L0 = K in the paper)
    tau = L // L0 # number of subframes (τ = L / L0)
    d = max(tau, (n_ris + 1)) # d = max(τ, N+1)
    q_dft = dft(d) # Generate the DFT matrix of size d x d
    Q = q_dft[0:n_ris + 1, 0:tau]  # Truncate to the first τ rows and first N+1 columns
    return Q

def generate_orthogonal_pilots(n_ue, L0):
    # Generate the DFT matrix of size L0 x L0
    dft_matrix = dft(L0)
    
    # Slice to get the first n_ue columns (pilot sequences for the users)
    pilots = dft_matrix[:, :n_ue]
    
    # Return the transpose so each column is a pilot sequence
    return pilots.T

def pilot_transmission(pilots, combined_channel, n_bs, n_ue, L0):
    L0 = n_ue # number of symbols per subframe (L0 = K in the paper)
    tau = L // L0 # number of subframes (τ = L / L0)
    
    Y = np.zeros((n_bs, n_ue, tau), dtype=complex)
    
    for t in range(tau):
        for k in range(n_ue):
            pilots_k = pilots[:, k].reshape(1, L0) # pilot sequence for user k
            
            # Pilots go through channel 
            yk = combined_channel[k, 0, :].reshape(n_bs, 1) @ pilots_k
            
            Y[:, k, t] += np.sum(yk, axis=1)
        Y[:, :, t] += np.sqrt(noise_var/2) * (np.random.randn(n_bs, n_ue) + 1j * np.random.randn(n_bs, n_ue))
    return Y

def process_received_pilots(Y, pilots, n_ue, L):
    L0 = n_ue  # number of symbols per subframe (L0 = K in the paper)
    tau = L // L0  # number of subframes (τ = L / L0)
    
    F_hat = np.zeros((Y.shape[0], pilots.shape[0]), dtype=complex)  # n_bs x n_ue
    
    for k in range(n_ue):  # Loop over each user
        Y_k = Y[:, k, :]  # Extract the Y matrix corresponding to user k
        
        # Perform the estimation for user k
        for t in range(tau):
            Y_t = Y_k[:, t]  # Y_t is (n_bs,)
            pilot_k_conj = np.conj(pilots[k, :])  # Get the pilot sequence for user k
            
            # Element-wise multiplication across the whole sequence
            F_hat[:, k] += Y_t * pilot_k_conj[t % L0]
    
    F_hat /= tau
    return F_hat


## System Setup
# Object Size
n_bs = 8 # Number of BS antennas
n_irs = 100 # Number of IRS elements 
n_ue = 3 # NUmber of users
n_uet = 1 # Number of antennas per user 
# Power, signal configuration (in dBm)
Pt_up = 15  # Uplink transmit power
Pt_down = 20  # Downlink transmit power
Po_up = -100  # Uplink noise power
Po_down = -65  # Downlink Noise Power 
epsilon = 10  # Rician factor 
lambda_c = 1  # Relative wavelengt
noise_var = (10**((Po_up)/10)) * 1e-3 
# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements - presumed
# Object Locations
bs_loc = np.array([100, -100, 0])  # BS location
irs_loc = np.array([0, 0, 0])  # IRS center location
# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements - presumed

# Pilot configuration 
pilot_lengths = [10, 20]  # Example pilot lengths to test

## Simulation 

for L in pilot_lengths: # testing alongt the range of pilot_lengths 
    print(f"Testing with pilot length: {L}")


    # Generate locations of users and IRS elements 
    ue_loc = generate_user_locations(n_ue)  # Generate user locations 
    irs_positions = generate_irs_positions(irs_loc, n_rows, n_cols, d_irs)  # Generate IRS positions
    
    d_bu = np.zeros(n_ue)  # Direct channel distance storage 
    
    for k in range(n_ue):     
        d_bu[k] = calculate_distance(bs_loc, ue_loc[k])  # Direct channel distance calculation 
    
    d_bi = calculate_distance(bs_loc, irs_loc)  # Distance BS-IRS 
    
    d_iu = np.zeros(n_ue)  # Distance IRS-UE storage
    
    for k in range(n_ue):
        d_iu[k] = calculate_distance(irs_loc, ue_loc[k])  # Distance IRS-UE
        
    # Path losses
    pl_bu = path_loss_direct(d_bu)  # direct channel path loss 
    pl_bi = path_loss_cascaded(d_bi)  # BS-IRS channel path loss
    pl_iu = path_loss_cascaded(d_iu)  # IRS-UE channel path loss 
    pl_cascaded = pl_bi + pl_iu  # total cascaded channel path loss
    
    pl_bu_linear = 10 ** (-pl_bu / 10)  # Linear scale direct channel path loss 
    
           
    # Channel coefficients calculation
    hkd = direct_channel(n_ue, n_uet, n_bs, pl_bu_linear) # Direct Channel
    hkr = IRS_UE_channel(n_ue, n_uet, n_irs, pl_iu, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols) # Compute IRS-to-UE channel matrix hkr
    # Compute the full BS-IRS channel matrix G
    d_bs = 0.5  # Assuming BS antenna spacing is lambda_c / 2
    G = BS_IRS_channel(n_irs, n_bs, pl_bi, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols)
    
    # Combine hkr and G
    combined_channel = np.zeros(hkd.shape, dtype=complex)
    for k in range(hkd.shape[0]):  # Loop over users
        combined_channel[k, 0, :] = G @ hkr[k, 0, :]
        
    phase_shifts = generate_phase_shifts(L, n_ue, n_irs)
    pilots = generate_orthogonal_pilots(n_ue, L)
    Y = pilot_transmission(pilots, combined_channel, n_bs, n_ue, L)

    F_hat = process_received_pilots(Y, pilots, n_ue, L)
