import numpy as np

n_bs = 8 # number of BS antennas
n_irs = 100 # no. of IRS elements
n_ue = 3 # no. of users

# Object Locations 
bs_loc = np.array([100, -100, 0]) # BS location 
irs_loc = np.array([0, 0, 0]) # IRS center location 

# User locations 
ue_x = np.random.uniform(5, 35, n_ue)
ue_y = np.random.uniform(-35, 35, n_ue)
ue_z = np.full(n_ue, -20)

ue_loc = np.vstack((ue_x,ue_y,ue_z)).T
