import numpy as np
import pandas as pd
import csv




def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
 
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]
        
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

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

    
   








def data_transform(future_guided, future_opt, data, n_feature, n_his, n_pred, time_len, time_start, device):

    
    day = 288
    week = day * 7
    
    
    
    n_vertex = data.shape[1] 
    len_record = len(data) 
    num = len_record - n_his - n_pred# 
    
    residual_samples_lm = []
    residual_samples = []
    F_E = []
    RR = []
    for i in range(len_record):
        index = i % week
        if future_opt is True:
            if index == 0:
                k_l = len(range(week-1, len_record, week))
                data_transfer_l = data[range(week-1, len_record, week), :, :]
            else:
                k_l = len(range(index - 1, len_record, week))
                data_transfer_l = data[range(index - 1, len_record, week), :, :]
            k_m = len(range(index, len_record, week))
            data_transfer_m = data[range(index, len_record, week), :, :]
            k_r = len(range((index + 1) % week, len_record, week))
            data_transfer_r = data[range((index + 1) % week, len_record, week), :, :]
            data_current = data[i-1] + data[i] + data[(i+1) % len_record]
            data_future = data[(i+1) % len_record]
            K_e = k_l + k_m + k_r - 3
            K_t = k_l + k_m + k_r
            data_transfer_sum = np.sum(data_transfer_l, axis=0) + np.sum(data_transfer_m, axis=0) + np.sum(data_transfer_r, axis=0)
        
        else:
            K = len(range(index, len_record, week))
            K_e = K - 1
            K_t = K
            data_current = data[i]
            
            data_transfer_sum = np.sum(data[range(index, len_record, week), :, :], axis=0)
        data_F_e = (data_transfer_sum - data_current) / K_e
        data_F_t = (data_transfer_sum) / K_t
        residual_sample = data_F_t - data_F_e
        rr = np.absolute(data[i] - data_F_e)
        F_E.append(data_F_e[:,0])
        RR.append(rr)
        residual_samples.append(residual_sample)
        if future_opt is True:
            data_F_t_lm = (data_transfer_sum - data_future) / (K_t - 1)
            residual_sample_lm = data_F_t_lm - data_F_e
        
            residual_samples_lm.append(residual_sample_lm)

        
        
    residual_samples_lm = np.array(residual_samples_lm)
    residual_samples = np.array(residual_samples)
    F_E = np.array(F_E)
    RR = np.array(RR)
        
        
    if future_guided is True:
        x = np.zeros([num, n_feature + n_feature + n_feature + 1, n_his, n_vertex])
    else:
        x = np.zeros([num, n_feature, n_his, n_vertex])
    y = np.zeros([num, n_pred, n_vertex])
    for i in range(num):
        head = i
        tail = i + n_his
        if future_guided is True:
            if future_opt is True:
                x[i,:n_feature,:-1] = residual_samples[head:tail-1].transpose(2, 0, 1)
                x[i,:n_feature,-1,:] = residual_samples_lm[tail-1].transpose(1, 0)
            else:
                x[i,:n_feature] = data[head:tail].transpose(2, 0, 1)
                x[i,n_feature:n_feature + n_feature] = residual_samples[head:tail].transpose(2, 0, 1)
                
            x[i,n_feature + n_feature:n_feature + n_feature + n_feature,:,:] = RR[head:tail].transpose(2, 0, 1)
            x[i,-1,:,:] = F_E[tail:tail + n_pred]
        else:
            x[i] = data[head:tail].transpose(2, 0, 1)
        # y_data:
        y[i] = data[tail: tail + n_pred, :, 0]

        
        
                
    
    return x, y