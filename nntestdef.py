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

