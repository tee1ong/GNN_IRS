import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Define GNN layers and model
class GraphLayer(layers.Layer):
    def __init__(self, output_dim):
        super(GraphLayer, self).__init__()
        self.dense = layers.Dense(output_dim, activation='relu')

    def call(self, inputs):
        return self.dense(inputs)

class GNN(Model):
    def __init__(self, output_dim):
        super(GNN, self).__init__()
        self.graph_layer1 = GraphLayer(output_dim)
        self.graph_layer2 = GraphLayer(output_dim)
    
    def call(self, inputs):
        x = self.graph_layer1(inputs)
        x = self.graph_layer2(x)
        return x

# Define GNN parameters
output_dim = 64  # Dimension of the output of each graph layer
initial_learning_rate = 1e-3
decay_rate = 0.98
decay_steps = 300
num_epochs = 100  # Training epochs
batch_size = 1024  # Number of samples per gradient update

# Define optimizer and learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Dummy input data for demonstration purposes (replace with actual data)
num_samples = 1024 * 10  # Example total number of training samples
input_dim = 10  # Example input dimension
x_train = np.random.randn(num_samples, input_dim)
y_train = np.random.randn(num_samples, output_dim)

# Create GNN model
gnn_model = GNN(output_dim)

# Compile model with loss and optimizer
gnn_model.compile(optimizer=optimizer, loss='mse')

# Train the model
gnn_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)

# Evaluate the model
loss = gnn_model.evaluate(x_train, y_train)
print(f"Final training loss: {loss}")
