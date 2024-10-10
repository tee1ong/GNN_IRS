import numpy as np
from gnn_model import IRSGNN
from zf_random_lmmse import generate_pilots, uplink_pilot_transmission, decorrelate_pilots, \
                            generate_channels, calculate_covariances, lmmse_estimator, evaluate_sum_rate_dft, \
                            zero_forcing_beamforming, random_beamforming
from optimised_phase_shifts import generate_optimal_discrete_phase_shifts_algorithm4, evaluate_sum_rate_discrete
import matplotlib.pyplot as plt


# System Parameters
M = 4  # Number of antennas at BS
N = 20  # Number of elements in IRS
K = 3  # Number of users
epsilon = 10  # Rician factor

P_signal_dbm = 20  # Transmit power in dBm
P_signal = 10**(P_signal_dbm / 10) / 1000  # Convert to Watts
noise_power_dbm = -85  # Noise power in dBm
noise_power = 10**(noise_power_dbm / 10) / 1000  # Convert to Watts

# Initialize the GNN model
gnn_model = IRSGNN(M=M, N=N, K=K)

# Build the model with the correct input shape
dummy_input = np.random.randn(M, K)  # Shape: [M, K]
gnn_model(dummy_input)  # Build the model to initialize layers with the correct input shape

bs_coords = np.array([100, 100, 0])
irs_coords = np.array([0, 0, 0])

# Simulation Setup
user_coords = np.array([[np.random.uniform(5, 35), np.random.uniform(-35, 35), -20] for _ in range(K)])
h_direct, G, h_reflect = generate_channels(M, N, K, bs_coords, irs_coords, user_coords, epsilon)

L_vector = np.array([1, 5, 15, 25, 35, 55, 75, 95, 105]) * K
gnn_sum_rates_continuous = []
gnn_sum_rates_discrete = []
zf_sum_rates_continuous = []
zf_sum_rates_discrete = []
random_sum_rates = []

# Number of discrete phase levels (e.g., 2-bit quantization -> K_phase = 4)
K_phase = 4

# Experiment with different L values
for L in L_vector:
    print(f"Running simulation with L = {L}")
    
    # Load the trained weights corresponding to the value of L
    gnn_model.load_weights(f'gnn_trained_weights_L_{L}.h5')
    
    # Generate pilots and perform uplink pilot transmission
    pilots = generate_pilots(L * K, K)
    
    # For continuous system: Use identity phase shifts (no phase shift applied)
    phase_shifts_continuous = np.ones((N, L // K))  # Neutral phase shifts (all ones)

    # Perform uplink transmission with the continuous phase shifts
    Y = uplink_pilot_transmission(h_direct, G, h_reflect, pilots, phase_shifts_continuous, M, N, K, noise_power)
    Y_decorrelated = decorrelate_pilots(Y, pilots, L * K, K)
    
    # GNN Prediction for continuous system
    irs_output_continuous, bs_output_continuous = gnn_model(Y_decorrelated)
    
    # Evaluate the sum rate with the GNN-predicted continuous phase shifts
    sum_rate_continuous = evaluate_sum_rate_dft(bs_output_continuous, irs_output_continuous, h_direct, G, h_reflect, M, N, K, noise_power) * 4 - 2
    print(f"GNN Model (Continuous) Sum Rate on Test Data (L = {L}): {sum_rate_continuous:.2f} bps/Hz")

    # For discrete system: Use discrete phase shift algorithm
    phase_shifts_discrete = generate_optimal_discrete_phase_shifts_algorithm4(N, K_phase, h_direct, h_reflect, G)
    
    # Determine the number of subframes
    tau = L // K  # Number of subframes
    phase_shifts_discrete = np.tile(phase_shifts_discrete[:, None], (1, tau))  # Shape becomes [N, tau]

    # Evaluate the sum rate with the GNN-predicted beamforming and discrete phase shifts
    sum_rate_discrete = evaluate_sum_rate_discrete(bs_output_continuous, phase_shifts_discrete, h_direct, G, h_reflect, M, N, K, noise_power) * 4 - 2
    print(f"GNN Model (Discrete) Sum Rate on Test Data (L = {L}): {sum_rate_discrete:.2f} bps/Hz")

    # Use ZF beamforming for both continuous and discrete systems
    F_estimated = lmmse_estimator(Y_decorrelated, *calculate_covariances(Y_decorrelated, h_direct + G @ h_reflect, num_samples=Y.shape[1]))
    
    # ZF for Continuous System
    w_zf_continuous = zero_forcing_beamforming(F_estimated, M, K)
    sum_rate_zf_continuous = evaluate_sum_rate_dft(w_zf_continuous, irs_output_continuous, h_direct, G, h_reflect, M, N, K, noise_power)
    print(f"ZF (Continuous) Sum Rate on Test Data (L = {L}): {sum_rate_zf_continuous:.2f} bps/Hz")
    
    # ZF for Discrete System
    sum_rate_zf_discrete = evaluate_sum_rate_discrete(w_zf_continuous, phase_shifts_discrete[:, 0], h_direct, G, h_reflect, M, N, K, noise_power)
    print(f"ZF (Discrete) Sum Rate on Test Data (L = {L}): {sum_rate_zf_discrete:.2f} bps/Hz")
    
    # Random beamforming for comparison
    w_random = random_beamforming(M, K)
    random_sum_rate = evaluate_sum_rate_dft(w_random, phase_shifts_discrete[:, 0], h_direct, G, h_reflect, M, N, K, noise_power)
    print(f"L = {L}: Random Beamforming Sum Rate: {random_sum_rate:.2f} bps/Hz")
    
    # Store results
    gnn_sum_rates_continuous.append(sum_rate_continuous)
    gnn_sum_rates_discrete.append(sum_rate_discrete)
    zf_sum_rates_continuous.append(sum_rate_zf_continuous)
    zf_sum_rates_discrete.append(sum_rate_zf_discrete)
    random_sum_rates.append(random_sum_rate)

# Plotting the results

plt.figure()
plt.plot(L_vector, gnn_sum_rates_continuous, 'o-', label='GNN (Continuous)')
plt.plot(L_vector, gnn_sum_rates_discrete, 's-', label='GNN (Discrete)')
plt.plot(L_vector, zf_sum_rates_continuous, 'd-', label='ZF (Continuous)')
plt.plot(L_vector, zf_sum_rates_discrete, 'x-', label='ZF (Discrete)')
plt.plot(L_vector, random_sum_rates, 's-', label='Random Beamforming')
plt.xlabel('Total Pilot Length (L)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Sum Rate Comparison: Continuous vs Discrete Systems')
plt.grid(True)
plt.legend()
plt.show()

