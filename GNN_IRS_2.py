import numpy as np

# Number of BS antennas, IRS elements, and users
n_bs = 8  # Number of BS antennas
n_irs = 100  # Number of IRS elements
n_ue = 3  # Number of users
n_uet = 1  # Number of antennas per user

# Power, signal config (in dBm)
Pt_up = 15  # Uplink transmit power
Pt_down = 20  # Downlink transmit power
Po_up = -100  # Uplink noise power
Po_down = -65  # Downlink Noise Power 
epsilon = 10  # Rician factor 
lambda_c = 1  # Relative wavelength

# Object Locations
bs_loc = np.array([100, -100, 0])  # BS location
irs_loc = np.array([0, 0, 0])  # IRS center location

# IRS configuration
n_rows = 10  # Number of rows in the IRS panel
n_cols = 10  # Number of columns in the IRS panel
d_irs = 0.5  # Distance between IRS elements - presumed

# Function definitions

# Uplink Pilot Transmission

# Generate orthogonal pilot sequences
def generate_orthogonal_pilots(n_ue, L0):
    pilots = np.fft.fft(np.eye(L0))[:n_ue].T  # Using DFT to generate orthogonal sequences
    return pilots

# Simulate the uplink pilot transmission and received signal at BS
def simulate_pilot_transmission(pilots, combined_channel, n_bs, L0, tau, noise_var):
    Y = np.zeros((n_bs, L0 * tau), dtype=complex)  # Received signal matrix at BS
    
    for t in range(tau):
        for k in range(n_ue):
            # Get the pilot sequence for user k
            pilot_sequence = pilots[:, k].reshape(1, L0)
            
            # Apply the combined channel to the pilot sequence
            received_signal = combined_channel[k, 0, :].reshape(n_bs, 1) @ pilot_sequence
            
            # Add the received signal to the corresponding part of Y
            Y[:, t*L0:(t+1)*L0] += received_signal
        
        # Add noise to the received signal
        Y[:, t*L0:(t+1)*L0] += np.sqrt(noise_var/2) * (np.random.randn(n_bs, L0) + 1j * np.random.randn(n_bs, L0))
    
    return Y

# Process the received pilots at the BS to estimate the combined channel
def process_received_pilots(Y, pilots, L0, tau):
    F_hat = np.zeros((Y.shape[0], pilots.shape[1]), dtype=complex)  # n_bs x n_ue
    
    for t in range(tau):
        Y_t = Y[:, t*L0:(t+1)*L0]
        
        for k in range(pilots.shape[1]):
            F_hat[:, k] += np.dot(Y_t, np.conj(pilots[:, k]))
    
    F_hat /= tau
    return F_hat

def lmmse_estimation(Y, pilots, noise_var, combined_channel_covariance):
    """
    Perform LMMSE channel estimation.

    Y: Received pilot matrix at BS.
    pilots: Transmitted pilot sequences.
    noise_var: Noise variance.
                                                                
    combined_channel_covariance: Covariance matrix of the combined channel.

    Returns:
    F_hat_lmmse: LMMSE estimate of the channel matrix.
    """

    Q = pilots.T @ pilots
    sigma2 = noise_var
    
    # Ensure the covariance matrix is of the correct shape
    R_y = combined_channel_covariance + sigma2 * np.eye(combined_channel_covariance.shape[0])
    
    # Expectation calculations (assuming Gaussian distribution)
    E_Y = np.mean(Y, axis=1, keepdims=True)  # Mean of Y (along the sub-frames)
    
    # LMMSE estimation
    F_hat_lmmse = np.dot(np.dot(combined_channel_covariance, np.linalg.inv(R_y)), Y - E_Y) + E_Y
    
    return F_hat_lmmse

# Downlink Beamforming

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

# User locations
def generate_user_locations(n_ue):
    x = np.random.uniform(5, 35, n_ue)  # x coordinate for users randomly distributed in a plane 
    y = np.random.uniform(-35, 35, n_ue)  # y coordinate for users randomly distributed in a plane
    z = np.full(n_ue, -20)  # constant z coordinate for the users
    return np.vstack((x, y, z)).T

# IRS Locations
def generate_irs_positions(irs_loc, n_rows, n_cols, d_irs):
    y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs)  # RIS elements y coordinates (horizontal)
    z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs)  # RIS elements z coordinates (vertical)
    y, z = np.meshgrid(y_coords, z_coords)  # y_coords x z_coords
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, irs_loc[0])  # x plane at 0 
    return np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T  # combine into an array of IRS element positions

# Direct channel path loss model 
def path_loss_direct(d):
    pl = 32.6 + 36.7 * np.log10(d)
    return pl

# Cascaded channel path loss model
def path_loss_cascaded(d):
    pl = 30 + 22 * np.log10(d)
    return pl

# Euclidean distance calculation 
def calculate_distance(a, b): 
    dist = np.linalg.norm(b - a)
    return dist

# Generates direct channel coefficients
def direct_channel(n_ue, n_uet, n_bs, beta_0):
    if len(beta_0) != n_ue:
        raise ValueError(f"Incompatible transmit power vectors")
    
    h = np.zeros((n_ue, n_uet, n_bs), dtype=complex)
    
    for k in range(n_ue):
        r_part = np.random.randn(n_uet, n_bs)
        i_part = np.random.randn(n_uet, n_bs)
        h[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * beta_0[k]
        
    return h

# IRS-to-UE channel matrix calculation
def IRS_UE_channel(n_ue, n_uet, n_irs, beta_1, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols):
    if len(beta_1) != n_ue:
        raise ValueError("Incompatible beta_1 vector size with the number of users")
    
    rician_factor_los = np.sqrt(epsilon / (1 + epsilon))
    rician_factor_nlos = np.sqrt(1 / (1 + epsilon))
    
    hkr_nlos = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    hkr_los = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    
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

# Simulation loop for different pilot lengths
pilot_lengths = [10, 20, 30, 40]  # Example pilot lengths to test

for L0 in pilot_lengths:
    print(f"Testing with pilot length: {L0}")
    
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

    # Direct Channel
    hkd = direct_channel(n_ue, n_uet, n_bs, pl_bu_linear)

    # Compute IRS-to-UE channel matrix hkr
    hkr = IRS_UE_channel(n_ue, n_uet, n_irs, pl_iu, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols)

    # Compute the full BS-IRS channel matrix G
    d_bs = 0.5  # Assuming BS antenna spacing is lambda_c / 2
    g = BS_IRS_channel(n_irs, n_bs, pl_bi, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols)

    combined_channel = np.zeros(hkd.shape, dtype=complex)
    for k in range(hkd.shape[0]):  # Loop over users
        combined_channel[k, 0, :] = g @ hkr[k, 0, :]

    # Simulate Uplink Pilot Transmission
    tau = n_bs + 1  # Number of sub-frames
    pilots = generate_orthogonal_pilots(n_ue, L0)  # Generate orthogonal pilots
    noise_var = 1e-3  # Noise variance

    # Simulate Uplink Pilot Transmission
    Y = simulate_pilot_transmission(pilots, combined_channel, n_bs, L0, tau, noise_var)
    
    # Estimate the combined channel covariance matrix
    combined_channel_covariance = np.cov(combined_channel.reshape(n_bs, -1))
    
    # Process the received pilots to estimate the channel
    F_hat = process_received_pilots(Y, pilots, L0, tau)
    # LMMSE Estimation
    F_hat_lmmse = lmmse_estimation(Y, pilots, noise_var, combined_channel_covariance)
    
    print(f"Estimated Channel Matrix F_hat for L0={L0} using LMMSE estimation:\n", F_hat_lmmse)


       