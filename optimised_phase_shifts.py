import numpy as np
import tensorflow as tf
from zf_random_lmmse import generate_pilots, uplink_pilot_transmission, decorrelate_pilots, \
                            generate_channels, calculate_covariances, lmmse_estimator, evaluate_sum_rate_dft, \
                            zero_forcing_beamforming, random_beamforming

# System Parameters
M = 4  # Number of antennas at BS
N = 20  # Number of elements in IRS
K = 3  # Number of users (K refers to the number of users)
epsilon = 10  # Rician factor

P_signal_dbm = 20  # Transmit power in dBm
P_signal = 10**(P_signal_dbm / 10) / 1000  # Convert to Watts
noise_power_dbm = -85  # Noise power in dBm
noise_power = 10**(noise_power_dbm / 10) / 1000  # Convert to Watts

# Function to generate optimal discrete phase shifts (Algorithm 4)
def generate_optimal_discrete_phase_shifts_algorithm4(N, K_phase, h_direct, h_reflect, G):
    """
    Generate optimal discrete phase shifts using Algorithm 4 from the second paper.
    
    Args:
    - N: Number of IRS elements
    - K_phase: Number of discrete phase shift levels (for quantization)
    - h_direct: Direct channel from BS to users (shape: [M, K])
    - h_reflect: Reflected channel from IRS to users (shape: [N, K])
    - G: IRS-BS channel (shape: [M, N])
    
    Returns:
    - Optimal discrete phase shifts for the IRS (Algorithm 4)
    """
    # Initialize variables
    omega = 2 * np.pi / K_phase  # Discrete phase shift step
    phase_set = np.arange(0, 2 * np.pi, omega)  # Discrete phase shift values
    optimal_phase_shifts = np.zeros(N)
    
    # Step 1: Initialize g with the direct channel contribution (shape: [M, K])
    g = np.copy(h_direct)
    
    # Step 2: Sort IRS elements by their channel magnitudes or properties to simplify updates
    indices_sorted = np.argsort(np.abs(h_reflect), axis=0).flatten()  # Sort IRS elements by channel magnitude
    h_reflect_sorted = h_reflect[indices_sorted]  # Sort h_reflect
    G_sorted = G[:, indices_sorted]  # Sort G by the same order as h_reflect
    
    # Step 3: Iterate over each IRS element
    for idx, n in enumerate(indices_sorted):
        best_phase = 0
        best_value = -np.inf
        
        # Step 4: Try all possible discrete phase shifts for element n
        for phase_shift in phase_set:
            rotated_channel = h_reflect_sorted[idx] * np.exp(1j * phase_shift)  # Shape: [K]
            
            # Step 5: Reflect through the IRS element and update g
            rotated_channel_reshaped = rotated_channel[None, :]  # Shape [1, K] to broadcast
            reflected_signal = G_sorted[:, idx:idx+1] @ rotated_channel_reshaped  # Shape: [M, K]
            test_g = g + reflected_signal  # Add the reflected signal to the total sum (shape [M, K])
            
            # Step 6: Compute the objective (e.g., SNR or sum rate maximization)
            value = np.sum(np.abs(test_g) ** 2)  # Sum over all elements
            
            # Step 7: Select the best phase shift
            if value > best_value:
                best_value = value
                best_phase = phase_shift
        
        # Step 8: Set the optimal phase shift for this element
        optimal_phase_shifts[n] = best_phase
        rotated_channel = h_reflect_sorted[idx] * np.exp(1j * best_phase)  # Shape: [K]
        rotated_channel_reshaped = rotated_channel[None, :]  # Shape [1, K] to broadcast
        g += G_sorted[:, idx:idx+1] @ rotated_channel_reshaped  # Update g with the selected phase shift
    
    return optimal_phase_shifts

# Load the channel model
bs_coords = np.array([100, 100, 0])
irs_coords = np.array([0, 0, 0])
user_coords = np.array([[np.random.uniform(5, 35), np.random.uniform(-35, 35), -20] for _ in range(K)])

# Generate the channels
h_direct, G, h_reflect = generate_channels(M, N, K, bs_coords, irs_coords, user_coords, epsilon)

# Number of pilot lengths to test
L_vector = np.array([1, 5, 15, 25, 35, 55, 75, 95, 105]) * K
gnn_sum_rates = []
zf_sum_rates = []
random_sum_rates = []

# Number of discrete phase levels (e.g., 2-bit quantization -> K_phase = 4)
K_phase = 4

# Testing different L values and generating phase shifts using Algorithm 4
def evaluate_sum_rate_discrete(w, v, h_direct, G, h_reflect, M, N, K, noise_power):
    """
    Evaluate the sum rate with the given beamforming vector w and phase shifts v.
    
    Args:
    - w: Beamforming matrix (shape: [M, K])
    - v: Phase shift matrix for IRS (shape: [N] or [N, tau], where tau is the number of subframes)
    - h_direct: Direct channel (shape: [M, K])
    - G: IRS-to-BS channel (shape: [M, N])
    - h_reflect: IRS-to-users channel (shape: [N, K])
    - M: Number of BS antennas
    - N: Number of IRS elements
    - K: Number of users
    - noise_power: Noise power
    
    Returns:
    - Sum rate (bps/Hz)
    """
    sum_rate = 0

    # Check if v is 1D or 2D
    if v.ndim == 1:
        # If v is 1D, assume there's only one subframe
        tau = 1
        v = v[:, np.newaxis]  # Reshape v to be 2D: [N, 1]
    else:
        tau = v.shape[1]  # Number of subframes

    # Iterate over each subframe/time slot and apply corresponding phase shifts
    for t in range(tau):
        # Get phase shifts for this subframe (1D array for subframe t)
        phase_shift_t = v[:, t]  # Shape: [N]

        # Compute combined channel for this subframe
        h_combined = h_direct + G @ np.diag(phase_shift_t) @ h_reflect

        # Compute received signal
        received_signal = h_combined.T @ w  # Shape: [K, K]

        # Signal and interference powers
        signal_power = np.abs(np.diag(received_signal)) ** 2
        interference_power = np.sum(np.abs(received_signal) ** 2, axis=0) - signal_power
        
        # Compute the sum rate for this subframe
        sum_rate += np.sum(np.log2(1 + signal_power / (interference_power + noise_power)))

    # Return the average sum rate across all subframes
    return sum_rate / tau


for L in L_vector:
    print(f"Running simulation with L = {L}")
    
    # Generate pilots and perform uplink pilot transmission
    pilots = generate_pilots(L * K, K)
    phase_shifts = generate_optimal_discrete_phase_shifts_algorithm4(N, K_phase, h_direct, h_reflect, G)
    
    # Determine the number of subframes based on the pilot length and number of users
    tau = L // K  # Number of subframes

    # Extend the phase shifts to match the number of subframes
    phase_shifts = np.tile(phase_shifts[:, None], (1, tau))  # Shape becomes [N, tau]

    # Pass the phase_shifts to the uplink_pilot_transmission function
    Y = uplink_pilot_transmission(h_direct, G, h_reflect, pilots, phase_shifts, M, N, K, noise_power)
    
    Y_decorrelated = decorrelate_pilots(Y, pilots, L * K, K)
    
    # Channel Estimation
    C_A, C_Y, mean_h, mean_Y = calculate_covariances(Y_decorrelated, h_direct + G @ h_reflect, num_samples=Y.shape[1])
    
    # Estimate channel using LMMSE
    F_estimated = lmmse_estimator(Y_decorrelated, C_A, C_Y, mean_h, mean_Y)
    
    # Use the ZF beamforming method
    w_zf = zero_forcing_beamforming(F_estimated, M, K)
    sum_rate_zf = evaluate_sum_rate_discrete(w_zf, phase_shifts, h_direct, G, h_reflect, M, N, K, noise_power)
    
    # Random beamforming for comparison
    w_random = random_beamforming(M, K)
    random_sum_rate = evaluate_sum_rate_discrete(w_random, phase_shifts, h_direct, G, h_reflect, M, N, K, noise_power)
    
    # Print and store results
    print(f"L = {L}: Zero-Forcing Beamforming Sum Rate: {sum_rate_zf:.2f} bps/Hz")
    print(f"L = {L}: Random Beamforming Sum Rate: {random_sum_rate:.2f} bps/Hz")
    
    zf_sum_rates.append(sum_rate_zf)
    random_sum_rates.append(random_sum_rate)

# Plotting the results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(L_vector, zf_sum_rates, 'o-', label='Zero-Forcing Beamforming')
plt.plot(L_vector, random_sum_rates, 's-', label='Random Beamforming')
plt.xlabel('Total Pilot Length (L)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Sum Rate vs Pilot Length (Discrete IRS Phase Shifts)')
plt.grid(True)
plt.legend()
plt.show()
