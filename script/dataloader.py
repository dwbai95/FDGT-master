import numpy as np
import pandas as pd
import csv


def get_adjacency_matrix(distance_df_filename, num_of_vertices):

    with open(distance_df_filename, "r") as f:
        reader = csv.reader(f)

        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    for i, j in edges:
        A[i, j], A[j, i] = 1, 1

    return A


def load_weighted_adjacency_matrix(file_path, n_vertex):

    df = get_adjacency_matrix(file_path, n_vertex)

    return df


def load(file_path):

    try:
        data = pd.read_csv(file_path, header=None)
        data = np.array(data)
    except:
        try:
            data = pd.read_hdf(file_path)
            data = np.array(data)
        except:
            try:
                data = np.load(file_path)
                data = data[data.files[0]]
            except:
                data = np.load(file_path)
    if data.ndim == 2:

        data = data.reshape(data.shape[0], data.shape[1], -1)

    return data


def data_transform(
    future_guided, data, n_feature, n_his, n_pred, time_len, time_start, device
):

    day = 288
    week = day * 7

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred  #

    residual_samples_lm = []
    residual_samples = []
    F_E = []
    F_T = []
    RR = []
    for i in range(len_record):
        index = i % week
        K = len(range(index, i + week, week))
        K_N = 4

        if K <= K_N:
            data_transfer_index = list(range(index, len_record, week))

            data_transfer_sum = np.sum(data[data_transfer_index[:K_N]], axis=0)
            K_e = K_N - 1
            K_t = K_N
            data_current = data[i]
            data_F_e = (data_transfer_sum - data_current) / K_e
            data_F_t = (data_transfer_sum) / K_t

        else:

            data_transfer_index = list(range(index + (K - K_N) * week, i + week, week))
            data_transfer_sum = np.sum(data[data_transfer_index], axis=0)
            K_e = K_N - 1
            K_t = K_N
            data_current = data[i]
            data_F_e = (data_transfer_sum - data_current) / K_e
            data_F_t = (data_transfer_sum) / K_t
        residual_sample = data_F_t - data_F_e
        rr = np.absolute(data[i] - data_F_e)
        F_E.append(data_F_e[:, 0])
        F_T.append(data_F_t[:, 0])
        RR.append(rr)
        residual_samples.append(residual_sample)

    residual_samples_lm = np.array(residual_samples_lm)
    residual_samples = np.array(residual_samples)
    F_E = np.array(F_E)
    F_T = np.array(F_T)
    RR = np.array(RR)

    if future_guided is True:
        x = np.zeros([num, 3 * n_feature + 1, n_his, n_vertex])
        y = np.zeros([num, n_pred, n_vertex])
    else:
        x = np.zeros([num, n_feature, n_his, n_vertex])
        y = np.zeros([num, n_pred, n_vertex])
    for i in range(num):
        head = i
        tail = i + n_his
        if future_guided is True:

            x[i, :n_feature] = data[head:tail].transpose(2, 0, 1)
            x[i, n_feature : n_feature + n_feature] = residual_samples[
                head:tail
            ].transpose(2, 0, 1)

            x[i, n_feature + n_feature : n_feature + n_feature + n_feature, :, :] = RR[
                head:tail
            ].transpose(2, 0, 1)
            x[i, -1, :, :] = F_E[tail : tail + n_pred]

        else:
            x[i] = data[head:tail].transpose(2, 0, 1)

        y[i] = data[tail : tail + n_pred, :, 0]

    return x, y
