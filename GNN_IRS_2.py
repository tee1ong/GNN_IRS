import numpy as np 
from scipy.linalg import dft

# Generate User Locations
def generate_user_locations(n):
    x = np.random.uniform(5, 35, n)
    y = np.random.uniform(-35, 35, n)
    z = np.full(n, -20)
    return np.vstack((x, y, z)).T

# Generate IRS Element Positions
def generate_irs_positions(irs_loc, n_rows, n_cols, d_irs):
    y_coords = np.arange(-(n_rows//2)*d_irs, (n_rows//2)*d_irs, d_irs)
    z_coords = np.arange(-(n_cols//2)*d_irs, (n_cols//2)*d_irs, d_irs)
    y, z = np.meshgrid(y_coords, z_coords)
    y = y.flatten()
    z = z.flatten()
    x = np.full_like(y, irs_loc[0])
    return np.vstack((x, y + irs_loc[1], z + irs_loc[2])).T

# Calculate Euclidean Distance
def calculate_distance(a, b): 
    return np.linalg.norm(b - a)

# Calculate Angles (phi, theta)
def calculate_angles(point_a, point_b):
    d_ab = calculate_distance(point_a, point_b)
    phi = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    theta = np.arctan2(point_b[2] - point_a[2], d_ab)
    return phi, theta

# Generate Steering Vector
def steering_vector(phi, theta, n, d_element, lambda_c, array_type='irs', n_cols=None):
    if array_type == 'irs':
        if n_cols is None:
            raise ValueError("n_cols must be provided for IRS steering vector calculation.")
        i1 = n % n_cols
        i2 = n // n_cols
        exponent = 2j * np.pi * d_element / lambda_c * (i1 * np.sin(phi) * np.cos(theta) + i2 * np.sin(theta))
    elif array_type == 'bs':
        exponent = 2j * np.pi * n * d_element / lambda_c * np.cos(phi) * np.cos(theta)
    else:
        raise ValueError("Invalid array_type. Choose 'irs' or 'bs'.")
    return np.exp(exponent)

# Generate Direct Channel
def direct_channel(n_ue, n_uet, n_bs, beta_0, num_samples=100):
    h = np.zeros((num_samples, n_ue, n_uet, n_bs), dtype=complex)
    for sample in range(num_samples):
        for k in range(n_ue):
            r_part = np.random.randn(n_uet, n_bs)
            i_part = np.random.randn(n_uet, n_bs)
            h[sample, k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * beta_0[k]
    return h

# Generate IRS-UE Channel
def IRS_UE_channel(n_ue, n_uet, n_irs, beta_1, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols, num_samples=100):
    hkr = np.zeros((num_samples, n_ue, n_uet, n_irs), dtype=complex)
    for sample in range(num_samples):
        rician_factor_los = np.sqrt(epsilon / (1 + epsilon))
        rician_factor_nlos = np.sqrt(1 / (1 + epsilon))

        hkr_nlos = np.zeros((n_ue, n_uet, n_irs), dtype=complex)
        hkr_los = np.zeros((n_ue, n_uet, n_irs), dtype=complex)

        for k in range(n_ue):
            r_part = np.random.randn(n_uet, n_irs)
            i_part = np.random.randn(n_uet, n_irs)
            hkr_nlos[k, :, :] = ((r_part + 1j * i_part) / np.sqrt(2)) * rician_factor_nlos

        for k in range(n_ue):
            phi_3_k, theta_3_k = calculate_angles(irs_loc, ue_loc[k])
            for m in range(n_uet):
                for n in range(n_irs):
                    hkr_los[k, m, n] = steering_vector(phi_3_k, theta_3_k, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols) * rician_factor_los

        for k in range(n_ue):
            hkr[sample, k, :, :] = beta_1[k] * (hkr_los[k, :, :] + hkr_nlos[k, :, :])

    return hkr

# Generate BS-IRS Channel
def BS_IRS_channel(n_irs, n_bs, beta_2, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols, num_samples=100):
    g_full = np.zeros((num_samples, n_bs, n_irs), dtype=complex)
    for sample in range(num_samples):
        phi_1, theta_1 = calculate_angles(bs_loc, irs_loc)
        g_los = np.zeros((n_bs, n_irs), dtype=complex)
        for n in range(n_irs):
            a_bs = steering_vector(phi_1, theta_1, n, d_bs, lambda_c, array_type='bs')
            phi_2, theta_2 = calculate_angles(irs_loc, bs_loc)
            a_irs = steering_vector(phi_2, theta_2, n, d_irs, lambda_c, array_type='irs', n_cols=n_cols)
            g_los[:, n] = np.outer(a_bs, a_irs.conj()).flatten()

        g_nlos = generate_nlos_component(n_bs, n_irs)
        g_full[sample, :, :] = beta_2 * (np.sqrt(epsilon / (1 + epsilon)) * g_los + np.sqrt(1 / (1 + epsilon)) * g_nlos)

    return g_full

# Generate NLOS Component for G
def generate_nlos_component(n_bs, n_irs):
    return (np.random.randn(n_bs, n_irs) + 1j * np.random.randn(n_bs, n_irs)) / np.sqrt(2)

# Path Loss Models

# Direct Channel Path Loss Model
def path_loss_direct(d):
    return 32.6 + 36.7 * np.log10(d)

# Cascaded Channel Path Loss Model
def path_loss_cascaded(d):
    return 30 + 22 * np.log10(d)

## Uplink Pilot Transmission

# Generate Orthogonal Pilot Sequences
def generate_phase_shifts(L, n_ue, n_ris):
    L0 = n_ue
    tau = L // L0
    d = max(tau, (n_ris + 1))
    q_dft = dft(d)
    Q = q_dft[0:n_ris + 1, 0:tau]
    return Q

def generate_orthogonal_pilots(n_ue, L):
    dft_matrix = dft(L)
    pilots = dft_matrix[:, :n_ue]
    return pilots


def pilot_transmission(pilots, combined_channel, n_bs, n_ue, L):
    L0 = pilots.shape[0]  # number of symbols per subframe (L0 = length of each pilot sequence)
    tau = L // L0  # number of subframes
    
    Y = np.zeros((n_bs, n_ue, tau), dtype=complex)
    
    for t in range(tau):
        for k in range(n_ue):
            pilots_k = pilots[:, k].reshape(L0, 1)  # pilot sequence for user k, shape (L0, 1)
            
            # Pilots go through the channel
            yk = combined_channel[:, k, 0, :].reshape(-1, n_bs, 1) @ pilots_k.T  # Adjusted to handle multiple samples
            
            Y[:, k, t] += np.sum(yk, axis=2).mean(axis=0)  # Average over samples for each subframe
        
        Y[:, :, t] += np.sqrt(noise_var / 2) * (np.random.randn(n_bs, n_ue) + 1j * np.random.randn(n_bs, n_ue))
    
    return Y


def lmmse_estimation(Y, pilots, noise_var, combined_channel_covariance):
    n_bs, n_ue, tau = Y.shape
    Y_reshaped = Y.reshape(n_bs, n_ue * tau)
    R_Y = combined_channel_covariance + noise_var * np.eye(combined_channel_covariance.shape[0])
    E_Y = np.mean(Y_reshaped, axis=1, keepdims=True)
    F_hat_lmmse = np.dot(np.dot(combined_channel_covariance, np.linalg.inv(R_Y)), (Y_reshaped - E_Y)) + E_Y
    F_hat_lmmse = F_hat_lmmse.reshape(n_bs, n_ue, tau)
    return F_hat_lmmse

def calculate_mse(F_hat_lmmse, combined_channel):
    combined_channel_mean = np.mean(np.transpose(combined_channel, (0, 3, 1, 2)), axis=0)  # Average over samples
    mse = np.mean(np.abs(F_hat_lmmse - combined_channel_mean) ** 2)
    return mse

def normalize_channel(channel):
    norm_factor = np.linalg.norm(channel)
    if norm_factor > 0:
        return channel / norm_factor
    return channel

## System Setup and Simulation
n_bs = 8
n_irs = 100
n_ue = 3
n_uet = 1
Pt_up = 15
Pt_down = 20
Po_up = -100
Po_down = -65
epsilon = 10
lambda_c = 1
noise_var = (10**((Po_up)/10)) * 1e-3
n_rows = 10
n_cols = 10
d_irs = 0.5
bs_loc = np.array([100, -100, 0])
irs_loc = np.array([0, 0, 0])
pilot_lengths = [10, 20]
num_samples = 100  # Number of samples

mse_errors = []  # List to store MSE for each pilot length

for L in pilot_lengths:
    print(f"Testing with pilot length: {L}")
    ue_loc = generate_user_locations(n_ue)
    irs_positions = generate_irs_positions(irs_loc, n_rows, n_cols, d_irs)
    
    d_bu = np.zeros(n_ue)
    for k in range(n_ue):
        d_bu[k] = calculate_distance(bs_loc, ue_loc[k])
    
    d_bi = calculate_distance(bs_loc, irs_loc)
    d_iu = np.zeros(n_ue)
    for k in range(n_ue):
        d_iu[k] = calculate_distance(irs_loc, ue_loc[k])
    
    pl_bu = path_loss_direct(d_bu)
    pl_bi = path_loss_cascaded(d_bi)
    pl_iu = path_loss_cascaded(d_iu)
    pl_cascaded = pl_bi + pl_iu
    
    pl_bu_linear = 10 ** (-pl_bu / 10)
    
    hkd = direct_channel(n_ue, n_uet, n_bs, pl_bu_linear, num_samples)
    hkd = normalize_channel(hkd)  # Normalize the direct channel

    hkr = IRS_UE_channel(n_ue, n_uet, n_irs, pl_iu, epsilon, irs_loc, ue_loc, d_irs, lambda_c, n_cols, num_samples)
    hkr = normalize_channel(hkr)  # Normalize the IRS-UE channel
    
    d_bs = 0.5
    G = BS_IRS_channel(n_irs, n_bs, pl_bi, epsilon, bs_loc, irs_loc, d_bs, d_irs, lambda_c, n_cols, num_samples)
    G = normalize_channel(G)  # Normalize the BS-IRS channel
    
    combined_channel = np.zeros(hkd.shape, dtype=complex)
    for k in range(hkd.shape[1]):
        combined_channel[:, k, 0, :] = G @ hkr[:, k, 0, :]
    
    phase_shifts = generate_phase_shifts(L, n_ue, n_irs)
    pilots = generate_orthogonal_pilots(n_ue, L)
    Y = pilot_transmission(pilots, combined_channel, n_bs, n_ue, L)
    combined_channel_covariance = np.cov(combined_channel.reshape(n_bs, -1))
    F_hat_lmmse = lmmse_estimation(Y, pilots, noise_var, combined_channel_covariance)
    
    mse = calculate_mse(F_hat_lmmse, combined_channel)
    mse_errors.append(mse)
    print(f"Mean Squared Error for L={L}: {mse}")

print(f"Final MSE errors across different pilot lengths: {mse_errors}")



       