# gnn_model.py
import tensorflow as tf
from tensorflow.keras import layers

class IRSGNN(tf.keras.Model):
    def __init__(self, M, N, K, hidden_dim=512, num_layers=2):
        super(IRSGNN, self).__init__()
        self.M = M
        self.N = N
        self.K = K

        # Input layers for user nodes and IRS node
        # Ensure input shape remains consistent without unexpected expansion
        self.user_input_layer = layers.Dense(1024, activation='relu')  # Input: [M, K] -> Output: [M, 1024]
        self.irs_input_layer = layers.Dense(1024, activation='relu')  # Input: [1, K] -> Output: [1, 1024]

        # GNN Layers for message passing
        self.gnn_layers_user = [layers.Dense(hidden_dim, activation='relu') for _ in range(num_layers)]
        self.gnn_layers_irs = [layers.Dense(hidden_dim, activation='relu') for _ in range(num_layers)]

        # Output layers
        self.irs_output_layer = layers.Dense(self.N, activation='linear')  # IRS phase shifts
        self.bs_output_layer = layers.Dense(self.K, activation='linear')  # Beamforming matrix

    def call(self, Y_decorrelated):
        # Input should remain [M, K] without unexpected transformations
        user_rep = self.user_input_layer(Y_decorrelated)  # Output: [M, 1024]

        # Process IRS inputs (using mean aggregation from users)
        irs_rep = tf.reduce_mean(Y_decorrelated, axis=0)  # Shape: [K] = [3]
        irs_rep = tf.expand_dims(irs_rep, axis=0)  # Shape: [1, K] = [1, 3]
        irs_rep = self.irs_input_layer(irs_rep)  # Output: [1, 1024]

        # GNN Layers for message passing (users and IRS)
        for user_gnn_layer, irs_gnn_layer in zip(self.gnn_layers_user, self.gnn_layers_irs):
            user_rep = user_gnn_layer(user_rep)  # Output from user GNN layer
            irs_rep = irs_gnn_layer(irs_rep)  # Output from IRS GNN layer

        # Final IRS and beamforming outputs
        irs_reflection = self.irs_output_layer(irs_rep)  # Shape: [1, N]
        irs_reflection = tf.reshape(irs_reflection, (self.N,))  # Reshape to [N]
        
        bs_beamforming = self.bs_output_layer(user_rep)  # Shape: [M, K]

        return irs_reflection, bs_beamforming
