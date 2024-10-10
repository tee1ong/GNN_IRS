import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt


# System Parameters
M = 4  # Number of antennas at BS
N = 20  # Number of elements in IRS (10x10)
K = 3  # Number of users
epsilon = 10  # Rician factor

P_signal_dbm = 20  # Transmit power in dBm
P_signal = 10**(P_signal_dbm / 10) / 1000  # Convert to Watts
noise_power_dbm = -85  # Noise power in dBm
noise_power = 10**(noise_power_dbm / 10) / 1000  # Convert to Watts

bs_coords = np.array([100, 100, 0])
irs_coords = np.array([0, 0, 0])

def path_loss_model(distance, model_type='direct'):
    if model_type == 'direct':
        return 10 ** ((32.6 + 36.7 * np.log10(distance)) / -10)
    elif model_type == 'rician':
        return 10 ** ((30 + 22 * np.log10(distance)) / -10)

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def steering_vector_irs(phi_irs, theta_irs, N):
    n_indices = np.arange(N)
    d_irs = 0.5  # Normalized by wavelength
    array_factor = d_irs * (n_indices % 10) * np.sin(theta_irs) * np.cos(phi_irs) + (n_indices // 10) * np.sin(theta_irs) * np.sin(phi_irs)
    return np.exp(1j * 2 * np.pi * array_factor)

def steering_vector_bs(phi_bs, theta_bs, M):
    m_indices = np.arange(M)
    d_bs = 0.5  # Normalized by wavelength
    array_factor = d_bs * m_indices * np.cos(theta_bs)
    return np.exp(1j * 2 * np.pi * array_factor)

def generate_channels(M, N, K, bs_coords, irs_coords, user_coords, epsilon):
    h_direct = np.zeros((M, K), dtype=complex)
    h_reflect = np.zeros((N, K), dtype=complex)
    G = np.zeros((M, N), dtype=complex)
    
    for k in range(K):
        # Direct channel from BS to user k
        d_bu = calculate_distance(bs_coords, user_coords[k])
        pl_direct = path_loss_model(d_bu, 'direct')
        h_direct[:, k] = np.sqrt(pl_direct) * (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)
        
        # IRS-user channel with Rician fading
        d_iu = calculate_distance(irs_coords, user_coords[k])
        pl_irs_user = path_loss_model(d_iu, 'rician')
        phi_irs_k = np.arctan2(user_coords[k][1] - irs_coords[1], user_coords[k][0] - irs_coords[0])
        theta_irs_k = np.arcsin((user_coords[k][2] - irs_coords[2]) / d_iu)
        h_r_los = steering_vector_irs(phi_irs_k, theta_irs_k, N).flatten()
        h_r_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        h_reflect[:, k] = np.sqrt(pl_irs_user) * (np.sqrt(epsilon/(1 + epsilon)) * h_r_los + np.sqrt(1/(1 + epsilon)) * h_r_nlos)

    # IRS-BS channel with Rician fading
    d_bi = calculate_distance(bs_coords, irs_coords)
    pl_irs_bs = path_loss_model(d_bi, 'rician')
    phi_bs = np.arctan2(irs_coords[1] - bs_coords[1], irs_coords[0] - bs_coords[0])
    theta_bs = np.arcsin((irs_coords[2] - bs_coords[2]) / d_bi)
    G_los = steering_vector_bs(phi_bs, theta_bs, M).reshape(M, 1) @ steering_vector_irs(phi_bs, theta_bs, N).reshape(1, N)
    G_nlos = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)
    G = np.sqrt(pl_irs_bs) * (np.sqrt(epsilon/(1 + epsilon)) * G_los + np.sqrt(1/(1 + epsilon)) * G_nlos)
    
    return h_direct, G, h_reflect

def generate_phase_shifts(N, tau):
    d = max(tau, N + 1)
    Q_full = dft(d)
    if tau <= N + 1:
        Q = Q_full[:, :tau]  
    else:
        Q = Q_full[:N + 1, :]  
    phase_shifts = Q / np.abs(Q)
    if phase_shifts.shape[0] > N:
        phase_shifts = np.delete(phase_shifts, 0, axis=0)
    return phase_shifts

def generate_pilots(L, K):
    L0 = K  
    tau = L // L0
    pilots_subframe = dft(L0)
    pilots_subframe = pilots_subframe[:, :K]  
    pilots = np.array([pilots_subframe] * tau)
    pilots = np.reshape(pilots, [L, K])
    return pilots

def uplink_pilot_transmission(h_direct, G, h_reflect, pilots, phase_shifts, M, N, K, noise_power):
    L = pilots.shape[0]
    tau = phase_shifts.shape[1]
    Y = np.zeros((M, L), dtype=complex)
    
    for l in range(L):
        v = phase_shifts[:, l % tau]
        h_combined = h_direct + G @ np.diag(v) @ h_reflect
        y_l = h_combined @ pilots[l, :] + np.sqrt(noise_power / 2) * (np.random.randn(M) + 1j * np.random.randn(M))
        Y[:, l] = y_l

    return Y

def decorrelate_pilots(Y, pilots, L, K):
    Y_decorrelated = np.zeros((Y.shape[0], K), dtype=complex)
    for k in range(K):
        pilot_k = pilots[:, k].reshape(L, 1)
        Y_decorrelated[:, k] = (Y @ np.conj(pilot_k)).flatten() / np.linalg.norm(pilot_k)**2
    return Y_decorrelated

def calculate_covariances(Y_decorrelated, true_channel, num_samples):
    mean_h = np.mean(true_channel, axis=0, keepdims=True)
    mean_Y = np.mean(Y_decorrelated, axis=0, keepdims=True)
    centered_true_channel = true_channel - mean_h
    centered_Y_decorrelated = Y_decorrelated - mean_Y
    C_A = (centered_true_channel.T @ centered_true_channel) / num_samples
    C_Y = (centered_Y_decorrelated.T @ centered_Y_decorrelated) / num_samples
    return C_A, C_Y, mean_h, mean_Y

def lmmse_estimator(Y_decorrelated, C_A, C_Y, mean_h, mean_Y):
    centered_Y = Y_decorrelated - mean_Y
    C_Y_inv = np.linalg.inv(C_Y)
    cross_covariance = C_A
    estimated_channel = (centered_Y @ C_Y_inv @ cross_covariance) + mean_h
    return estimated_channel

def zero_forcing_beamforming(h_combined, M, K):
    h_pseudo_inv = np.linalg.pinv(h_combined)
    w_zf = h_pseudo_inv.T
    w_zf /= np.linalg.norm(w_zf, axis=0, keepdims=True)
    return w_zf

def random_beamforming(M, K):
    w_random = np.random.randn(2 * M * K)
    w_random = w_random[:M * K].reshape(M, K) + 1j * w_random[M * K:].reshape(M, K)
    w_random /= np.linalg.norm(w_random, axis=0, keepdims=True)
    return w_random

def evaluate_sum_rate_dft(w, v, h_direct, G, h_reflect, M, N, K, noise_power):
    h_combined = h_direct + G @ np.diag(v) @ h_reflect
    received_signal = h_combined.T @ w
    signal_power = np.abs(np.diag(received_signal)) ** 2
    interference_power = np.sum(np.abs(received_signal) ** 2, axis=0) - signal_power
    sum_rate = np.sum(np.log2(1 + signal_power / (interference_power + noise_power)))
    return sum_rate

# Simulation Setup
user_coords = np.array([[np.random.uniform(5, 35), np.random.uniform(-35, 35), -20] for _ in range(K)])
h_direct, G, h_reflect = generate_channels(M, N, K, bs_coords, irs_coords, user_coords, epsilon)

L_vector = np.array([1,5,15,25,35,55,75,95,105]) * K
zf_sum_rates = []
random_sum_rates = []

# Experiment with different L values
for L in L_vector:
    print(f"Running simulation with L = {L}")
    pilots = generate_pilots(L * K, K)
    phase_shifts = generate_phase_shifts(N, L // K)  # Use the correct function to generate phase shifts

    Y = uplink_pilot_transmission(h_direct, G, h_reflect, pilots, phase_shifts, M, N, K, noise_power)
    Y_decorrelated = decorrelate_pilots(Y, pilots, L * K, K)

    # Channel Estimation
    C_A, C_Y, mean_h, mean_Y = calculate_covariances(Y_decorrelated, h_direct + G @ h_reflect, num_samples=Y.shape[1])

    # Estimate channel using LMMSE
    F_estimated = lmmse_estimator(Y_decorrelated, C_A, C_Y, mean_h, mean_Y)

    # Beamforming and Sum Rate Evaluation
    w_zf = zero_forcing_beamforming(F_estimated, M, K)
    w_random = random_beamforming(M, K)
    zf_sum_rate = evaluate_sum_rate_dft(w_zf, phase_shifts[:, 0], h_direct, G, h_reflect, M, N, K, noise_power)  
    random_sum_rate = evaluate_sum_rate_dft(w_random, phase_shifts[:, 0], h_direct, G, h_reflect, M, N, K, noise_power)
    
    zf_sum_rates.append(zf_sum_rate)
    random_sum_rates.append(random_sum_rate)

#     print(f"L = {L}: Zero-Forcing Beamforming Sum Rate: {zf_sum_rate:.2f} bps/Hz")
#     print(f"L = {L}: Random Beamforming Sum Rate: {random_sum_rate:.2f} bps/Hz")
#     print("\n")

# Plotting the results
plt.figure()
plt.plot(L_vector, zf_sum_rates, 'o-', label='Zero-Forcing Beamforming')
plt.plot(L_vector, random_sum_rates, 's-', label='Random Beamforming')
plt.xlabel('Total Pilot Length (L)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Sum Rate vs Pilot Length')
plt.grid(True)
plt.legend()
plt.show()

