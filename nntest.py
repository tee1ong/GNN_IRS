import numpy as np
import tensorflow as tf

# Hyperparameters
lr_initial = 1e-3 # Initial learning rate 
decay_rate = 0.98 
decay_steps = 300 # no. of iterations after whicch the learning rate is decreased
epochs = 100
it_per_epoch = 100 # iterations per epoch 
batch_size = 1024
epoch_limit = 10

# Define the learning rate schedule using ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = lr_initial,
    decay_steps = decay_steps,
    decay_rate = decay_rate,
    staircase = True  # Ensures the decay happens in discrete steps
)

# Define the Adam optimizer using the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)