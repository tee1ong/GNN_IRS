# train_gnn_discrete_sum_rate_vs_users.py
import numpy as np
import tensorflow as tf
from gnn_model import IRSGNN  # Import the GNN model
from zf_random_lmmse import generate_pilots, uplink_pilot_transmission, decorrelate_pilots, generate_channels

# System Parameters
M = 4  # Number of antennas at BS
N = 20  # Number of elements in IRS
epochs = 80  # Total number of epochs for training
num_iterations_per_epoch = 40  # Iterations per epoch
batch_size = 512  # Batch size for training
initial_learning_rate = 0.00001
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=300, decay_rate=0.98, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# Custom Loss Function (Sum Rate Maximization)
def custom_loss_function(irs_output, bs_output, h_direct, G, h_reflect, M, N, K, noise_power):
    irs_output_complex = tf.cast(irs_output, dtype=tf.complex128)
    bs_output_complex = tf.cast(bs_output, dtype=tf.complex128)

    h_combined = h_direct + tf.matmul(G, tf.linalg.diag(irs_output_complex)) @ h_reflect
    received_signal = tf.matmul(h_combined, tf.expand_dims(bs_output_complex, axis=-1))

    signal_power = tf.abs(tf.squeeze(received_signal)) ** 2
    interference_power = tf.reduce_sum(tf.abs(received_signal) ** 2, axis=0) - signal_power

    signal_power = tf.cast(signal_power, dtype=tf.float32)
    interference_power = tf.cast(interference_power, dtype=tf.float32)
    noise_power = tf.cast(noise_power, dtype=tf.float32)

    sum_rate = tf.reduce_sum(tf.math.log(1 + signal_power / (interference_power + noise_power + 1e-9)) / tf.math.log(2.0))
    return -sum_rate  # Negative for maximization

# Train GNN Model Over Different User Configurations
num_users_list = [1, 3, 5, 7, 10]  # Different number of users to test
sum_rates_vs_users = []

# Loop over different number of users (K)
for K in num_users_list:
    print(f"Training GNN for K = {K} users")
    
    # Initialize the GNN model for the current number of users
    gnn_model = IRSGNN(M=M, N=N, K=K)
    
    # Generate user coordinates and channels
    user_coords = np.array([[np.random.uniform(5, 35), np.random.uniform(-35, 35), -20] for _ in range(K)])
    h_direct, G, h_reflect = generate_channels(M, N, K, bs_coords=np.array([100, 100, 0]), irs_coords=np.array([0, 0, 0]), user_coords=user_coords, epsilon=10)

    # Train the GNN for each user configuration
    for epoch in range(epochs):
        epoch_sum_rate = 0  # To accumulate the sum rate for the epoch

        for iteration in range(num_iterations_per_epoch):
            # Generate pilots and perform uplink transmission
            L = 15 * K  # Example pilot length proportional to K
            pilots = generate_pilots(L, K)
            phase_shifts = np.random.uniform(0, 2 * np.pi, size=(N, L // K))  # Generate random phase shifts
            
            # Uplink transmission
            Y = uplink_pilot_transmission(h_direct, G, h_reflect, pilots, phase_shifts, M, N, K, noise_power=1e-9)
            Y_decorrelated = decorrelate_pilots(Y, pilots, L, K)
            
            with tf.GradientTape() as tape:
                irs_output, bs_output = gnn_model(Y_decorrelated)

                # Compute the loss (negative sum rate)
                loss = custom_loss_function(irs_output, bs_output, h_direct, G, h_reflect, M, N, K, noise_power=1e-9)

            gradients = tape.gradient(loss, gnn_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gnn_model.trainable_variables))

            # Accumulate sum rate (negating the loss gives the positive sum rate)
            epoch_sum_rate += -loss.numpy()

        # Average sum rate for the current epoch
        average_epoch_sum_rate = epoch_sum_rate / num_iterations_per_epoch
        print(f"K = {K}, Epoch {epoch + 1}, Average Sum Rate: {average_epoch_sum_rate:.2f} bps/Hz")

        # Save model weights after each epoch (optional)
        gnn_model.save_weights(f'gnn_trained_weights_K_{K}_epoch_{epoch}.h5')

    # After training, evaluate the GNN model
    # You can now run the evaluation function here if you want to compute the test sum rate
    # and store it in sum_rates_vs_users (similar to the testing script)
    # sum_rate = evaluate_sum_rate_discrete(bs_output, phase_shifts_discrete, h_direct, G, h_reflect, M, N, K, noise_power=1e-9)
    
# Optionally: Plot training sum rate vs epochs here if desired for each K configuration
# (Similar to how we plotted sum rates over different epochs in previous examples)
