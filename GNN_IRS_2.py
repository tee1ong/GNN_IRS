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

# Universal angle calculation function
def calculate_angles(point_a, point_b):
    """
    Calculate the azimuth and elevation angles between two points.
    
    point_a: The first point in space (e.g., BS or IRS location).
    point_b: The second point in space (e.g., IRS or UE location).
    
    Returns:
    - Azimuth angle (phi) and elevation angle (theta).
    """
    # Calculate the distance between the two points in 3D space
    d_ab = calculate_distance(point_a, point_b)
    
    # Azimuth angle
    phi = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    
    # Elevation angle
    theta = np.arctan2(point_b[2] - point_a[2], d_ab)
    
    return phi, theta

# Universal steering vector function
def steering_vector(phi, theta, n, d_element, lambda_c, array_type='irs', n_cols=None):
    """
    Calculate the steering vector for either IRS or BS.
    
    phi: Azimuth angle.
    theta: Elevation angle.
    n: Element index.
    d_element: Distance between array elements (e.g., d_irs or d_bs).
    lambda_c: Wavelength of the signal.
    array_type: 'irs' for IRS or 'bs' for BS. Defaults to 'irs'.
    n_cols: Number of columns in the IRS (required if array_type is 'irs').
    
    Returns:
    - Steering vector (complex exponential).
    """
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
        phi_3_k, theta_3_k = calculate_angles(irs_loc, ue_loc[k])
        for m in range(n_uet):  # For each user antenna
            for n in range(n_irs):  # For each IRS element
                hkr_los[k, m, n] = steering_vector(phi_3_k, theta_3_k, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols) * rician_factor_los
    
    # Combine the LOS and NLOS components, and apply path loss beta_1
    hkr = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
    for k in range(n_ue):
        hkr[k, :, :] = beta_1[k] * (hkr_los[k, :, :] + hkr_nlos[k, :, :])
    
    return hkr

# BS-IRS channel matrix calculation
def BS_IRS_channel(n_irs, n_bs, beta_2, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols):
    
    # Compute the LOS component
    phi_1, theta_1 = calculate_angles(bs_loc, irs_loc)
    g_los = np.zeros((n_bs, n_irs), dtype=complex)
    for n in range(n_irs):
        a_bs = steering_vector(phi_1, theta_1, n, d_bs, lambda_c, array_type='bs')
        phi_2, theta_2 = calculate_angles(irs_loc, bs_loc)
        a_irs = steering_vector(phi_2, theta_2, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols)
        g_los[:, n] = np.outer(a_bs, a_irs.conj()).flatten()
    
    # Generate the NLOS component
    g_nlos = generate_nlos_component(n_bs, n_irs)
    
    # Combine the LOS and NLOS components, and apply path loss beta_2
    g_full = beta_2 * (np.sqrt(epsilon / (1 + epsilon)) * g_los + np.sqrt(1 / (1 + epsilon)) * g_nlos)
    
    return g_full

# Function to generate NLOS component for G
def generate_nlos_component(n_bs, n_irs):
    # Generate the NLOS component as a complex Gaussian matrix
    return (np.random.randn(n_bs, n_irs) + 1j * np.random.randn(n_bs, n_irs)) / np.sqrt(2)

# Simulation

ue_loc = generate_user_locations(n_ue)  # Generate user locations 
irs_positions = generate_irs_positions(irs_loc, n_rows, n_cols, d_irs)  # Generate IRS positions

# Direct channel distance BS-UE
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


        