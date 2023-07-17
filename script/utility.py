import numpy as np
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power
from script import metrics 

import torch
import pandas as pd

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0] 
    id_mat = np.asmatrix(np.identity(n_vertex)) 
 
 
    # D_row
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))  
    # D_com
    #deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))

    # D = D_row as default
    deg_mat = deg_mat_row #deg
    adj_mat = np.asmatrix(adj_mat) 
    
    deg_mat_inv = np.linalg.inv(deg_mat) 
    deg_mat_inv[np.isinf(deg_mat_inv)] = 0. #

    deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
    deg_mat_inv_sqrt[np.isinf(deg_mat_inv_sqrt)] = 0.

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    wid_deg_mat_inv = np.linalg.inv(wid_deg_mat)
    wid_deg_mat_inv[np.isinf(wid_deg_mat_inv)] = 0.

    wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)
    wid_deg_mat_inv_sqrt[np.isinf(wid_deg_mat_inv_sqrt)] = 0.

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    # Symmetric normalized Laplacian
    # For SpectraConv
    # To [0, 1]
    # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
    sym_normd_lap_mat = id_mat - np.matmul(np.matmul(deg_mat_inv_sqrt, adj_mat), deg_mat_inv_sqrt)

    # For ChebConv
    # From [0, 1] to [-1, 1]
    # wid_L_sym = 2 * L_sym / lambda_max_sym - I
    #sym_max_lambda = max(np.linalg.eigvalsh(sym_normd_lap_mat))
    sym_max_lambda = max(eigvalsh(sym_normd_lap_mat))
    wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / sym_max_lambda - id_mat

    # For GCNConv
    # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
    hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

    # Random Walk normalized Laplacian
    # For SpectraConv
    # To [0, 1]
    # L_rw = D^{-1} * L_com = I - D^{-1} * A
    
    rw_normd_lap_mat = id_mat - np.matmul(deg_mat_inv, adj_mat)

    # For ChebConv
    # From [0, 1] to [-1, 1]
    # wid_L_rw = 2 * L_rw / lambda_max_rw - I
    #rw_max_lambda = max(np.linalg.eigvalsh(rw_normd_lap_mat))
    rw_max_lambda = max(eigvalsh(rw_normd_lap_mat))
    wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / rw_max_lambda - id_mat

    # For GCNConv
    # hat_L_rw = wid_D^{-1} * wid_A
    hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat
    elif mat_type == 'sym_normd_lap_mat':
        return sym_normd_lap_mat
    elif mat_type == 'ChebNet_sym':
        return wid_sym_normd_lap_mat
    elif mat_type == 'GCN_sym':
        return hat_sym_normd_lap_mat
    elif mat_type == 'rw_normd_lap_mat':
        return rw_normd_lap_mat
    elif mat_type == 'ChebNet_rw':
        return wid_rw_normd_lap_mat
    elif mat_type == 'GCN_rw':
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

def evaluate_model(model, loss, data_iter, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x.to(device)).reshape(y.shape)
            l = loss(y_pred, y.to(device))
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, zscore, n_pred, device):
    model.eval()
    with torch.no_grad():

        targets = []
        outs = []

        for x, y in data_iter:
            b,n,m = y.shape
            y = y.reshape(b*n,m)
            target_unnormalized = zscore.inverse_transform(y.cpu().numpy())
            target_unnormalized = target_unnormalized.reshape(b,n,m)
            target_unnormalized[target_unnormalized<0.1]=0
            y_pred = model(x.to(device))
        
            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred.reshape(b*n,m)
            
            out_unnormalized = zscore.inverse_transform(y_pred)
            out_unnormalized = out_unnormalized.reshape(b,n,m)
 
            targets.append(target_unnormalized)
            outs.append(out_unnormalized)
        

        out_unnormalized = np.concatenate(outs, axis=0).transpose(1,0,2)
        out_unnormalized[out_unnormalized<0]=0
        target_unnormalized = np.concatenate(targets, axis=0).transpose(1,0,2)
        mae=[]
        rmse=[]
        mape=[]
        rrse=[]
        corr=[]
        for i in range(12):
            MAE, MAPE, RMSE, RRSE, CORR = metrics.All_Metrics(pred=out_unnormalized[i], true=target_unnormalized[i], mask1 = None, mask2 = 0.)
            mae.append(MAE)
            mape.append(MAPE)
            rmse.append(RMSE)
            rrse.append(RRSE)
            corr.append(CORR)
            
        test_MAE, test_MAPE, test_RMSE, test_RRSE, test_CORR = metrics.All_Metrics(pred=out_unnormalized, true=target_unnormalized, mask1 = None, mask2 = 0.)
        
        print("Average Horizon, MAE: {:.2f}, MAPE: {:.2f}%, RMSE: {:.2f}".format(
                    test_MAE, test_MAPE, test_RMSE))
        
        mae.append(test_MAE)
        mape.append(test_MAPE)
        rmse.append(test_RMSE)
        rrse.append(test_RRSE)
        corr.append(test_CORR)
        
        accuracy_index = []
        accuracy_columns = ['MAE', 'MAPE', 'RMSE', 'RRSE', 'CORR']
        for i in range(12):
            accuracy_index.append('Horizon'+str(i+1))
        accuracy_index.append('Average Horizon')
        accuracy = np.round(np.array([mae, mape, rmse, rrse, corr]).T, 2)
        accuracy_pd = pd.DataFrame(accuracy, index = accuracy_index, columns = accuracy_columns)
        print(accuracy_pd)
        
        
        
        return accuracy_pd, out_unnormalized, target_unnormalized
