import numpy as np
import scipy.io as sio
from scipy.linalg import dft
import matplotlib.pyplot as plt


def generate_location(num_users):
    location_user = np.empty([num_users, 3]) # array for 3 users
    for k in range(num_users): # loop creates random x and y coordinates for the user, the z coordinate is set at -20
        x = np.random.uniform(5, 15)
        y = np.random.uniform(-15, 15)
        z = -20
        coordinate_k = np.array([x, y, z]) # Concatenated location coordinates for user k
        location_user[k, :] = coordinate_k # append to location_user
    return location_user


def path_loss_r(d):
    loss = 30 + 22.0 * np.log10(d)  # path loss model for BS-IRS and IRS-UE links
    return loss


def path_loss_d(d):
    loss = 32.6 + 36.7 * np.log10(d) # BS-user k link path loss model
    return loss


def generate_pathloss_aoa_aod(location_user, location_bs, location_irs):
    """
    :param location_user: array (num_user,2)
    :param location_bs: array (2,)
    :param location_irs: array (2,)
    :return: pathloss = (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user)
            cos_phi = (cos_phi_1, cos_phi_2, cos_phi_3)
    """

    num_user = location_user.shape[0] # (number of users, coordinates)
    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs) # calculates the euclidean distance BS-IRS
    pathloss_irs_bs = path_loss_r(d0)
    aoa_bs = ( location_irs[0] - location_bs[0]) / d0 # cos(phi1*)cos(theta1*)
    aod_irs_y = (location_bs[1]-location_irs[1]) / d0 # sin(phi2*)cos(theta2*)
    aod_irs_z = (location_bs[2]-location_irs[2]) / d0 # sin(theta2*)
    # =========irs-user=============
    pathloss_irs_user = []
    aoa_irs_y = []
    aoa_irs_z = []
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_irs) # euclidean distance user k - IRS
        pathloss_irs_user.append(path_loss_r(d_k)) # previous function
        aoa_irs_y_k = (location_user[k][1] - location_irs[1]) / d_k # sin(phi3,k*)cos(theta3,k*)
        aoa_irs_z_k = (location_user[k][2] - location_irs[2]) / d_k # sin(theta3,k*)
        aoa_irs_y.append(aoa_irs_y_k)
        aoa_irs_z.append(aoa_irs_z_k)
    aoa_irs_y = np.array(aoa_irs_y)
    aoa_irs_z = np.array(aoa_irs_z)

    # =========bs-user=============
    pathloss_bs_user = np.zeros([num_user, 1])
    for k in range(num_user):
        d_k = np.linalg.norm(location_user[k] - location_bs) # euclidean distance user k - BS (direct LOS channel)
        pathloss_bs_user_k = path_loss_d(d_k)
        pathloss_bs_user[k, :] = pathloss_bs_user_k

    pathloss = (pathloss_irs_bs, np.array(pathloss_irs_user), np.array(pathloss_bs_user)) # pathloss vector
    aoa_aod = (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) # angle vector (aoa - angle of arrival, aod - angle of departure)
    return pathloss, aoa_aod


def generate_channel(params_system, location_bs=np.array([100, -100, 0]), location_irs=np.array([0, 0, 0]),
                     location_user_initial=None, Rician_factor=10, scale_factor=100, num_samples=100,irs_Nh = 10):
    # scale_factor: can be viewed as (downlink noise_power_dB- downlink Pt)

    (num_antenna_bs, num_elements_irs, num_user) = params_system

    channel_bs_irs, channel_bs_user, channel_irs_user, set_location_user = [], [], [], []
    for ii in range(num_samples):
        if location_user_initial is None:
            location_user = generate_location(num_user)
            set_location_user.append(location_user)
        else:
            location_user = location_user_initial
            set_location_user.append(location_user)

        pathloss, aoa_aod = generate_pathloss_aoa_aod(location_user, location_bs, location_irs)
        (pathloss_irs_bs, pathloss_irs_user, pathloss_bs_user) = pathloss # unpack tuples into individual variables
        (aoa_bs, aod_irs_y, aod_irs_z, aoa_irs_y, aoa_irs_z) = aoa_aod

        pathloss_bs_user = pathloss_bs_user - scale_factor
        pathloss_irs_bs = pathloss_irs_bs - scale_factor / 2
        pathloss_irs_user = pathloss_irs_user - scale_factor / 2 # divide by 2 due to being half the power of LOS channel
        pathloss_bs_user = np.sqrt(10 ** ((-pathloss_bs_user) / 10)) # db to linear conversion
        pathloss_irs_user = np.sqrt(10 ** ((-pathloss_irs_user) / 10))
        pathloss_irs_bs = np.sqrt(10 ** ((-pathloss_irs_bs) / 10))

        # tmp:(num_antenna_bs,num_user) channel between BS and user - complex channel indexes  - LOS channel
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_user])
        tmp = tmp * pathloss_bs_user.reshape(1, num_user)
        channel_bs_user.append(tmp)

        # tmp: (num_antenna_bs,num_elements_irs) channel between IRS and BS  - NLOS channel
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, num_elements_irs])
        a_bs = np.exp(1j * np.pi * aoa_bs * np.arange(num_antenna_bs))
        a_bs = np.reshape(a_bs, [num_antenna_bs, 1])

        i1 = np.mod(np.arange(num_elements_irs),irs_Nh) # vertical IRS indices
        i2 = np.floor(np.arange(num_elements_irs)/irs_Nh) # horizontal IRS indices
        a_irs_bs = np.exp(1j * np.pi * (i1*aod_irs_y+i2*aod_irs_z)) #computes the phase contribution from horizontal and vertical indices.
        a_irs_bs =  np.reshape(a_irs_bs, [num_elements_irs, 1])
        los_irs_bs = a_bs @ np.transpose(a_irs_bs.conjugate()) # array LOS channel
        tmp = np.sqrt(Rician_factor / (1 + Rician_factor)) * los_irs_bs + np.sqrt(1/(1 + Rician_factor)) * tmp
        tmp = tmp * pathloss_irs_bs # combined channel  channel array
        channel_bs_irs.append(tmp)

        # tmp:(num_elements_irs,num_user) channel between IRS and user
        tmp = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs, num_user])
        for k in range(num_user):
            a_irs_user = np.exp(1j * np.pi * (i1 * aoa_irs_y[k] + i2 * aoa_irs_z[k]))
            tmp[:, k] = np.sqrt(Rician_factor/(1+Rician_factor))*a_irs_user+np.sqrt(1/(1+Rician_factor))*tmp[:, k]
            tmp[:, k] = tmp[:, k] * pathloss_irs_user[k]
        channel_irs_user.append(tmp)
    channels = (np.array(channel_bs_user), np.array(channel_irs_user), np.array(channel_bs_irs))
    return channels, set_location_user


def channel_complex2real(channels):
    channel_bs_user, channel_irs_user, channel_bs_irs = channels
    (num_sample, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape # Shape Extraction, determines the parameters inside the bracket
    num_user = channel_irs_user.shape[2] # No. of Users extraction

    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user]) # real values matrix
    # Hd_real = np.zeros([num_sample, 2 * num_antenna_bs, num_user])
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user], dtype=complex) #  Stores the combined channel matrix

    for kk in range(num_user):
        channel_irs_user_k = channel_irs_user[:, :, kk] #Extracts the channel matrix for user kk
        channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape(num_sample, 1, num_elements_irs) # Combines the BS-IRS and IRS-user channel matrices.
        set_channel_combine_irs[:, :, :, kk] = channel_combine_irs #Stores the combined channel matrix.
        A_tmp_tran = np.transpose(channel_combine_irs, (0, 2, 1)) # Transposes the combined channel matrix.
        A_tmp_real1 = np.concatenate([A_tmp_tran.real, A_tmp_tran.imag], axis=2) # Separates real and imaginary components
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag, A_tmp_tran.real], axis=2) # Further concatenates the real and imaginary parts to form the final real-valued matrix. # Read Section IV,C.
        A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
        A_T_real[:, :, :, kk] = A_tmp_real # Stores the real-valued matrix.

    Hd_real = np.concatenate([channel_bs_user.real, channel_bs_user.imag], axis=1) # Separates and combines real and imaginary components as individual elements

    return A_T_real, Hd_real, np.array(set_channel_combine_irs)

def combine_channel(channel_bs_user_k, channel_irs_user_k, channel_bs_irs, phase_shifts):
    channel_combine_irs = channel_bs_irs @ np.diag(phase_shifts)
    channel_combine = channel_bs_user_k + channel_combine_irs @ channel_irs_user_k
    # channel_combine_irs2 = channel_bs_irs @ np.diag(channel_irs_user_k) @  phase_shifts
    return channel_combine, channel_combine_irs


def batch_combine_channel(channel_bs_user_k, channel_irs_user_k, channel_bs_irs, phase_shifts):
    (num_sample, num_antenna_bs, num_elements_ir) = channel_bs_irs.shape
    len_pilots = phase_shifts.shape[1]

    channel_combine_irs = channel_bs_irs * channel_irs_user_k.reshape((num_sample, 1, num_elements_ir))
    channel_bs_user_k = np.repeat(channel_bs_user_k, len_pilots, axis=1)
    channel_combine = channel_bs_user_k.reshape((num_sample, num_antenna_bs, len_pilots)) \
                      + channel_combine_irs @ phase_shifts

    return channel_combine


def random_beamforming(num_test, num_antenna_bs, num_elements_irs, num_user):
    w_rnd = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_antenna_bs, num_user]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_antenna_bs, num_user]) # random beamforming weights, time function
    w_rnd_norm = np.linalg.norm(w_rnd, axis=(1, 2), keepdims=True) # Euclidean norm
    w_rnd = w_rnd / w_rnd_norm # normalization of the weights

    theta_rnd = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_elements_irs]) \
                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_test, num_elements_irs]) # random phase shifts
    theta_rnd = theta_rnd / np.abs(theta_rnd) # normalized phase shifts

    return w_rnd, theta_rnd


def ls_estimator(y, x): # estimates the channel matrix using the received signal y and the transmit signal x
    """
    y = h *x + n
    y: batch_size*m*l
    h: batch_size*m*n
    x: batch_size*n*l

    Output: h = y*x^H*(x*x^H)^-1
    """
    n, ell = x.shape[0], x.shape[1] # extracts number of columns and rows from transmit signal matrix x
    x_H = np.transpose(x.conjugate()) # Hermitian transpose of x
    if ell < n: # if number of columns is lower than number of rows
        x_Hx = np.matmul(x_H, x)
        # print('Cond number:',np.linalg.cond(x_Hx))
        x_Hx_inv = np.linalg.inv(x_Hx)
        h = np.matmul(y, x_Hx_inv)
        h = np.matmul(h, x_H)
    elif ell == n:  # number of columns equals number of rows
        # print('Cond number:',np.linalg.cond(x))
        h = np.linalg.inv(x)
        h = np.matmul(y, h)
    else:
        xx_H = np.matmul(x, x_H)
        # print('Cond number:',np.linalg.cond(xx_H))
        xx_H_inv = np.linalg.inv(xx_H)
        h = np.matmul(y, x_H)
        h = np.matmul(h, xx_H_inv)
    return h


def lmmse_estimator(Y, Q, C_A, C_Y, mean_A, mean_Y):
    # # Y = AQ+N

    # ================================================
    # A = np.matmul(Y,np.linalg.inv(C_Y))
    # A = np.matmul(A,np.transpose(Q.conjugate()))
    # A = np.matmul(A,C_A)

    Y = Y - mean_Y
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    n, ell = Q.shape[0], Q.shape[1]
    if ell > n:
        QQ_H = np.matmul(Q, Q_H)
        C_A_inv = np.linalg.inv(C_A)
        tmp = np.linalg.inv(gamma_n * C_A_inv + QQ_H)
        tmp = np.matmul(tmp, QQ_H)
        tmp = np.matmul(C_A_inv, tmp)
        tmp = np.matmul(tmp, C_A)
        A = ls_estimator(Y, Q)
        A = np.matmul(A, tmp)
    else:
        tmp = np.matmul(Q_H, C_A)
        tmp = np.matmul(tmp, Q)
        tmp = tmp + gamma_n * np.eye(ell)
        tmp = np.linalg.inv(tmp)
        A = np.matmul(Y, tmp)
        A = np.matmul(A, Q_H)
        A = np.matmul(A, C_A)

    return A + mean_A

def generate_pilots_bl(len_pilot, num_elements_irs, num_user): # Deterministic
    len_frame = num_user # L0
    num_frame = len_pilot // len_frame #tau
    if num_frame > num_elements_irs + 1:
        phase_shifts = dft(num_frame)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]
    else:
        phase_shifts = dft(num_elements_irs + 1)
        phase_shifts = phase_shifts[0:num_elements_irs + 1, 0:num_frame]

    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)
    phase_shifts = np.delete(phase_shifts, 0, axis=0)

    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, num_user])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(X)), X)), '\n')
    return phase_shifts, pilots



def generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user): # Random
    len_frame = num_user
    num_frame = len_pilot // len_frame
    phase_shifts = np.random.randn(num_elements_irs, num_frame) + \
                   1j * np.random.randn(num_elements_irs, num_frame)
    phase_shifts = phase_shifts / np.abs(phase_shifts)

    phase_shifts = np.repeat(phase_shifts, len_frame, axis=1)

    # pilots = dft(len_pilot)
    # pilots = pilots[:, 0:num_user]

    pilots_subframe = dft(len_frame)
    pilots_subframe = pilots_subframe[:, 0:num_user]
    pilots = np.array([pilots_subframe] * num_frame)
    pilots = np.reshape(pilots, [len_pilot, num_user])
    # print('X^H * X:\n ', np.diagonal(np.matmul(np.conjugate(np.transpose(pilots)), pilots)), '\n')
    return phase_shifts, pilots


def generate_received_pilots(channels, phase_shifts, pilots, noise_power_db, scale_factor=100, Pt=25):
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    (num_samples, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]
    len_pilots = phase_shifts.shape[1]

    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))

    y = []
    y_real = []

    for ii in range(num_samples):
        y_tmp = np.zeros((num_antenna_bs, len_pilots), dtype=complex)
        for ell in range(len_pilots):
            for k in range(num_user):
                channel_bs_user_k = channel_bs_user[ii, :, k]
                channel_irs_user_k = channel_irs_user[ii, :, k]
                channel_bs_irs_i = channel_bs_irs[ii]
                channel_combine, _ = combine_channel(channel_bs_user_k, channel_irs_user_k,
                                                     channel_bs_irs_i, phase_shifts[:, ell])
                pilots_k = pilots[:, k]
                y_tmp[:, ell] = y_tmp[:, ell] + channel_combine * pilots_k[ell]

        noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, len_pilots]) \
                + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_antenna_bs, len_pilots])
        y_tmp = y_tmp + noise_sqrt * noise
        y.append(y_tmp)
        y_tmp_real = np.concatenate([y_tmp.real, y_tmp.imag], axis=0)
        y_real.append(y_tmp_real)

    return np.array(y), np.array(y_real)


def generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db, scale_factor=100, Pt=15):
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    (num_samples, num_antenna_bs, num_elements_irs) = channel_bs_irs.shape
    num_user = channel_irs_user.shape[2]
    len_pilots = phase_shifts.shape[1]

    noise_sqrt = np.sqrt(10 ** ((noise_power_db - Pt + scale_factor) / 10))

    y = np.zeros((num_samples, num_antenna_bs, len_pilots), dtype=complex)
    for kk in range(num_user):
        channel_bs_user_k = channel_bs_user[:, :, kk]
        channel_irs_user_k = channel_irs_user[:, :, kk]
        channel_combine = batch_combine_channel(channel_bs_user_k, channel_irs_user_k,
                                                channel_bs_irs, phase_shifts)
        pilots_k = pilots[:, kk]
        pilots_k = np.array([pilots_k] * num_samples)
        pilots_k = pilots_k.reshape((num_samples, 1, len_pilots))
        y = y + channel_combine * pilots_k

    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots]) \
            + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_samples, num_antenna_bs, len_pilots])
    y = y + noise_sqrt * noise

    y_real = np.concatenate([y.real, y.imag], axis=1)

    return np.array(y), np.array(y_real)


def decorrelation(received_pilots, pilots):
    """
    Decorrelate the received pilot signals using the provided pilot sequences.
    
    Args:
        received_pilots (array): Received pilot signals, shape (num_samples, num_antenna_bs, len_pilots).
        pilots (array): The pilot sequences, shape (len_pilots, num_user).
    
    Returns:
        y_decode (array): Decorrelated received signals, shape (num_samples, num_antenna_bs, num_user, num_frame).
    """
    len_pilots, num_user = pilots.shape
    num_samples, num_antenna_bs, _ = received_pilots.shape
    pilots = np.array([pilots] * num_samples)
    pilots = pilots.reshape((num_samples, len_pilots, num_user))

    len_frame = num_user
    num_frame = len_pilots // len_frame

    x_tmp = np.conjugate(pilots[:, 0:len_frame, :])
    y_decode = np.zeros([num_samples, num_antenna_bs, num_user, num_frame], dtype=complex)
    
    for jj in range(num_frame):
        y_k = received_pilots[:, :, jj * len_frame:(jj + 1) * len_frame]
        y_decode_tmp = y_k @ x_tmp / len_frame
        y_decode[:, :, :, jj] = y_decode_tmp

    return y_decode


# Statistical info: mean, covariance, noise covariance
def compute_stat_info(params_system, noise_power_db, location_user, Rician_factor, num_samples=10000):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = num_user * 1
    len_frame = num_user
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)
    channels, set_location_user = generate_channel(params_system,location_user_initial=location_user,
                                                   Rician_factor=Rician_factor, num_samples=num_samples)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    y, _ = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
    Y = decorrelation(y, pilots)
    A, Hd, = channel_bs_irs_user, channel_bs_user

    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    A, Hd, Y = A[:, :, :, 0], Hd[:, :, 0], Y[:, :, 0, :]
    # A_h = np.zeros([num_samples, num_antenna_bs, num_elements_irs + 1]) + 1j * np.zeros(
    #     [num_samples, num_antenna_bs, num_elements_irs + 1])
    # for ii in range(num_samples):
    #     A_h[ii, :, :] = np.concatenate((Hd[ii, :].reshape(-1, 1), A[ii, :, :]), axis=1)
    A_h = np.concatenate((Hd.reshape(-1, num_antenna_bs, 1), A), axis=2)
    A = A_h

    mean_A, mean_Y = np.mean(A, axis=0, keepdims=True), np.mean(Y, axis=0, keepdims=True)
    # print(mean_Y - mean_A @ Q)
    A = A - mean_A
    C_A = np.sum(np.matmul(np.transpose(A.conjugate(), (0, 2, 1)), A), axis=0) / num_samples
    Y = Y - mean_Y
    # print(Y-A@Q)
    C_Y = np.sum(np.matmul(np.transpose(Y.conjugate(), (0, 2, 1)), Y), axis=0) / num_samples
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    stat_info = (gamma_n, C_A, mean_A)
    return stat_info


def channel_estimation_lmmse(params_system, y, pilots, phase_shifts, stat_info):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]

    len_frame = num_user
    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]

    (gamma_n, C_A, mean_A) = stat_info
    C_Y = np.matmul(np.matmul(np.transpose(Q.conjugate()), C_A), Q) + gamma_n * np.eye(Q.shape[1])
    mean_Y = np.matmul(mean_A, Q)

    y_d = decorrelation(y, pilots)
    channel_bs_user_est = np.zeros((num_sample, num_antenna_bs, num_user), dtype=complex)
    channel_bs_irs_user_est = np.zeros((num_sample, num_antenna_bs, num_elements_irs, num_user), dtype=complex)
    for kk in range(num_user):
        y_k = y_d[:, :, kk, :]

        channel_est = lmmse_estimator(y_k, Q, C_A, C_Y, mean_A, mean_Y)
        channel_bs_user_est[:, :, kk] = channel_est[:, :, 0]
        channel_bs_irs_user_est[:, :, :, kk] = channel_est[:, :, 1:num_elements_irs + 1]

    return channel_bs_user_est, channel_bs_irs_user_est


def test_channel_estimation_lmmse(params_system, len_pilot, noise_power_db, location_user, Rician_factor, num_sample):
    (num_antenna_bs, num_elements_irs, num_user) = params_system
    # phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)
    phase_shifts, pilots = generate_pilots_bl_v2(len_pilot, num_elements_irs, num_user)

    # print(phase_shifts, np.abs(phase_shifts))
    # print(pilots, '\n\n', np.diag(pilots @ np.transpose(pilots.conjugate())))
    channels, set_location_user = generate_channel(params_system,
                                                   num_samples=num_sample, location_user_initial=location_user,
                                                   Rician_factor=Rician_factor)
    (channel_bs_user, channel_irs_user, channel_bs_irs) = channels
    _, _, channel_bs_irs_user = channel_complex2real(channels)
    # y1, y1_r = generate_received_pilots(channels, phase_shifts, pilots, noise_power_db)
    y, y_real = generate_received_pilots_batch(channels, phase_shifts, pilots, noise_power_db)
    stat_info = compute_stat_info(params_system, noise_power_db, location_user, Rician_factor)

    # ===channel estimation===
    channel_bs_user_est, channel_bs_irs_user_est = channel_estimation_lmmse(params_system, y, pilots, phase_shifts,stat_info)

    #---MSE---
    # err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2
    # err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2
    #---NMSE---
    err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2/np.linalg.norm(channel_bs_user, axis=(1))**2
    err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2/np.linalg.norm(channel_bs_irs_user, axis=(1,2))**2

    # print('Direct link estimation error (num_sample, num_user):\n', err_bs_user)
    # print('Cascaded link estimation error (num_sample, num_user):\n', err_bs_irs_user)
    return np.mean(err_bs_user), np.mean(err_bs_irs_user)



num_antenna_bs, num_elements_irs, num_user, num_sample = 8, 100, 3, 100
params_system = (num_antenna_bs, num_elements_irs, num_user)
noise_power_db, Rician_factor = -100, 5
location_user = None
set_len_pilot = np.array([1,5,15,25,35,55,75,95,105])*num_user
len_frame = num_user
num_samples = 100
err_lmmse = []


channels, set_location_user = generate_channel(params_system,
                                               num_samples=num_sample, location_user_initial=location_user,
                                               Rician_factor=Rician_factor)
(channel_bs_user, channel_irs_user, channel_bs_irs) = channels
_, _, channel_bs_irs_user = channel_complex2real(channels)

    
for len_pilot in set_len_pilot:
    
    phase_shifts, pilots = generate_pilots_bl(len_pilot, num_elements_irs, num_user)
    y, y_real = generate_received_pilots(channels, phase_shifts, pilots, noise_power_db)
    Y = decorrelation(y, pilots)
    A, Hd, = channel_bs_irs_user, channel_bs_user
    ones = np.ones((1, len_pilot))
    phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    Q = phaseshifts_new[:, 0:len_pilot:len_frame]
    A, Hd, Y = A[:, :, :, 0], Hd[:, :, 0], Y[:, :, 0, :]
    A_h = np.concatenate((Hd.reshape(-1, num_antenna_bs, 1), A), axis=2)
    A = A_h
    
    mean_A, mean_Y = np.mean(A, axis=0, keepdims=True), np.mean(Y, axis=0, keepdims=True)
    # print(mean_Y - mean_A @ Q)
    A = A - mean_A
    C_A = np.sum(np.matmul(np.transpose(A.conjugate(), (0, 2, 1)), A), axis=0) / num_samples
    Y = Y - mean_Y
    C_Y = np.sum(np.matmul(np.transpose(Y.conjugate(), (0, 2, 1)), Y), axis=0) / num_samples
    Q_H = np.transpose(Q.conjugate())
    C_N = C_Y - np.matmul(Q_H, np.matmul(C_A, Q))
    gamma_n = np.real(np.mean(np.diagonal(C_N)))
    stat_info = (gamma_n, C_A, mean_A)
   
    
    len_pilot = pilots.shape[0]
    num_sample = y.shape[0]

    # len_frame = num_user
    # # ones = np.ones((1, len_pilot))
    # # phaseshifts_new = np.concatenate([ones, phase_shifts], axis=0)
    # # Q = phaseshifts_new[:, 0:len_pilot:len_frame]
    C_Y = np.matmul(np.matmul(np.transpose(Q.conjugate()), C_A), Q) + gamma_n * np.eye(Q.shape[1])
    mean_Y = np.matmul(mean_A, Q)

    y_d = decorrelation(y, pilots)
    channel_bs_user_est = np.zeros((num_sample, num_antenna_bs, num_user), dtype=complex)
    channel_bs_irs_user_est = np.zeros((num_sample, num_antenna_bs, num_elements_irs, num_user), dtype=complex)
    for kk in range(num_user):
        y_k = y_d[:, :, kk, :]

        channel_est = lmmse_estimator(y_k, Q, C_A, C_Y, mean_A, mean_Y)
        channel_bs_user_est[:, :, kk] = channel_est[:, :, 0]
        channel_bs_irs_user_est[:, :, :, kk] = channel_est[:, :, 1:num_elements_irs + 1]
    # channel_bs_user_est, channel_bs_irs_user_est = channel_estimation_lmmse(params_system, y, pilots, phase_shifts,stat_info)
    
    err_bs_user = np.linalg.norm(channel_bs_user_est - channel_bs_user, axis=(1))**2/np.linalg.norm(channel_bs_user, axis=(1))**2
    err_bs_irs_user = np.linalg.norm(channel_bs_irs_user_est - channel_bs_irs_user, axis=(1, 2))**2/np.linalg.norm(channel_bs_irs_user, axis=(1,2))**2
    err3, err4  = np.mean(err_bs_user), np.mean(err_bs_irs_user)



    # err3, err4 = test_channel_estimation_lmmse(params_system, len_pilot, noise_power_db, location_user, Rician_factor,
                                                # num_sample)
    # print('ls estimation:', err1, err2)
    print('lmmse estimation:', err3, err4)
    err_lmmse.append(err3+err4)
print(err_lmmse)

#----------
plt.figure()
plt.title('Error')
plt.plot(set_len_pilot,err_lmmse,'s-',label='lmmse')
plt.xlabel('Pilot length')
plt.ylabel('MSE')
plt.grid()
plt.show()


