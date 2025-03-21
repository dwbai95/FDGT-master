import logging    
import os   
import argparse    
import configparser    
import math 
import random 

import numpy as np
import pandas as pd
from sklearn import preprocessing 
import sys
import datetime
import time
import torch
import torch.nn as nn

import torch.optim as optim
import torch.utils as utils
#import pynvml


from script import dataloader, utility, earlystopping 
from model import models

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"



class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
     
        self.log = open(filename, "a", encoding="utf-8")  
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
    def reset(self):
        self.log.close()
        sys.stdout=self.terminal

def set_seed(seed):          
    os.environ['PYTHONHASHSEED']=str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = True 

def worker_init_fn(worker_id):
    set_seed(worker_id)  
    
     
def get_parameters(dataset, gpu_index): 
    parser = argparse.ArgumentParser(description='FDGT for road traffic prediction')
    parser.add_argument('--enable_cuda', type=bool, default = True,
                        help='enable CUDA to_csv fault as True')

    parser.add_argument('--n_pred', type=int, default=12, 
                        help='the number of time interval for predcition, default as 3') 
    parser.add_argument('--epochs', type=int, default=300,
                        help='epochs, default as 200') 
    parser.add_argument('--dataset_config_path', type=str, default='./FDGT_master_0/config/'+dataset+'.ini',
                        help='the path of dataset config file, pemsd7-m.ini for PeMSD7-M, \
                            metr-la.ini for METR-LA, and pems-bay.ini for PEMS-BAY')   
    parser.add_argument('--opt', type=str, default='AdamW',
                        help='optimizer, default as AdamW')
    args = parser.parse_args(args=[]) 

    print('--------Training configs----------')
    for k in list(vars(args).keys()):      
        print('%s: %s' % (k, vars(args)[k]))
    

    config = configparser.ConfigParser() 

    

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device("cuda:"+str(gpu_index))
    else:
        device = torch.device("cpu")
        
    enable_cuda = args.enable_cuda
    dataset_config_path = args.dataset_config_path

    config.read(dataset_config_path, encoding="utf-8")
    

    def ConfigSectionMap(section):  
        dict1 = {} 
        options = config.options(section) 
        for option in options:
            try:
                dict1[option] = config.get(section, option) 
                if dict1[option] == -1:
                    logging.debug("skip: %s" % option) 
            except:
                print("exception on %s!" % option) 
                dict1[option] = None 
        return dict1
    
    for k in ConfigSectionMap('data'):          
        print(k, ':', ConfigSectionMap('data')[k])
    
    print('--------Training configs----------\n')
    dataset = ConfigSectionMap('data')['dataset']

    n_vertex = int(ConfigSectionMap('data')['n_vertex'])
    n_feature = int(ConfigSectionMap('data')['n_feature'])
    time_intvl = int(ConfigSectionMap('data')['time_intvl'])
    time_len = int(ConfigSectionMap('data')['time_len'])
    time_start = ConfigSectionMap('data')['time_start']
    n_his = int(ConfigSectionMap('data')['n_his'])
    Kt = int(ConfigSectionMap('data')['kt']) 
    stblock_num = int(ConfigSectionMap('data')['stblock_num'])
    if (         (Kt - 1) * 2 * stblock_num > n_his) or ((Kt - 1) * 2 * stblock_num <= 0): 
        raise ValueError(f'ERROR: {Kt} and {stblock_num} are unacceptable.')
    
    day_slot = int(24 * 60 / time_intvl)
    drop_rate = float(ConfigSectionMap('data')['drop_rate']) # drop_out_rate
    learning_rate = float(ConfigSectionMap('data')['learning_rate'])
    weight_decay_rate = float(ConfigSectionMap('data')['weight_decay_rate']) 
    step_size = int(ConfigSectionMap('data')['step_size']) 
    gamma = float(ConfigSectionMap('data')['gamma']) 
    data_path = ConfigSectionMap('data')['data_path'] 
    
    
    wam_path = ConfigSectionMap('data')['wam_path'] 
    
   
    n_pred = args.n_pred # 
    

    
    
    data = dataloader.load(data_path)
    
    adj_mat = dataloader.load_weighted_adjacency_matrix(wam_path, n_vertex) 


    opt = args.opt
    

    return device,  n_his, n_pred, day_slot, time_intvl, time_len, time_start, data, adj_mat, stblock_num, n_feature, n_vertex, drop_rate, opt, learning_rate, weight_decay_rate, step_size, gamma, enable_cuda





def data_preparate(future_guided, future_opt, data, device, n_his, n_pred, day_slot, time_len, time_start, batch_size):
    
    data_col, n_vertex, n_feature = data.shape
    
    val_rate = 0.2
    test_rate = 0.2

    
    
    len_val = int(math.floor(data_col * val_rate)) 
    len_test = int(math.floor(data_col * test_rate))
    len_train = int(data_col - len_val -len_test)
        
        
        
    zscore = []
    zscore_flow = preprocessing.StandardScaler() 
    zscore_lane = preprocessing.StandardScaler()
    zscore_speed = preprocessing.StandardScaler()
    zscore.append(zscore_flow)
    zscore.append(zscore_lane)
    zscore.append(zscore_speed)


    data_z = np.zeros([data_col, n_vertex, n_feature])
    
        
    for i in range(n_feature):
        data_z[:,:,i] = zscore[i].fit_transform(data[:,:,i]) 
    
    x_data, y_data = dataloader.data_transform(future_guided, data_z, n_feature, n_his, n_pred, time_len, time_start, device)
    time_index = pd.date_range(start=time_start,  periods = time_len, freq='5min', name=None)
    data_col = len(x_data)
    len_val = int(math.floor(data_col * val_rate)) 
    len_test = int(math.floor(data_col * test_rate)) 
    len_train = int(data_col - len_val -len_test) 
    time_index_test = time_index[-(n_pred - 1 + len_test):]
    
    x_train = x_data[: len_train] 
    x_val = x_data[len_train: len_train + len_val]
    x_test = x_data[len_train + len_val:]

    y_train = y_data[: len_train] 
    y_val = y_data[len_train: len_train + len_val]
    y_test = y_data[len_train + len_val:] 
 
    
    train_data = utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False) 
    
    return zscore[0], train_iter, val_iter, test_iter, data, time_index_test



def model_main(future_guided, graph_conv_type, adj_mat, mat_type, order, Kt, Ks, batch_size, stblock_num, n_feature, n_his, n_pred, n_vertex, gated_act_func, drop_rate, device, learning_rate, weight_decay_rate, model_save_path, step_size, train_patience,  gamma, opt):
    
     
    
    
    blocks = []
    if future_guided is True:
        blocks.append([3 * n_feature * (1 + order)])
        blocks.append([64, 16, 64])
    else:
        blocks.append([n_feature * (1 + order)])
        blocks.append([64, 16 , 64])
    blocks.append([128, 128])
    blocks.append([n_pred])
        
    mat = utility.calculate_laplacian_matrix(adj_mat, mat_type) 
    chebconv_matrix = torch.from_numpy(mat).float().to(device) 


    model = models.FDGT(future_guided, order, Kt, Ks, blocks, batch_size, n_feature, n_his, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate, device).to(device) 
    
   
    loss = nn.MSELoss()

    learning_rate = learning_rate
    weight_decay_rate = weight_decay_rate
    early_stopping = earlystopping.EarlyStopping(patience=train_patience, path=model_save_path, verbose=True)
    

    if opt == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=20)
    return model, loss, early_stopping, optimizer, scheduler

def train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter, enable_cuda, n_pred, device):
 
    min_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    epoch_list = []
    Inference_time_list = []
    Training_time_list = []
    
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  
        model.train()
        start_training = time.time()
        for x, y in train_iter: 
           
            y_pred = model(x.to(device)) 

          
            l = loss(y_pred, y.to(device))
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            
        train_loss = l_sum / n
        end_training = time.time()
        
        Training_time_list.append(end_training-start_training)
        start_inference = time.time()
        
        val_loss = val(model, val_iter, n_pred, device)
        
        end_inference = time.time()
        
        Inference_time_list.append(end_inference-start_inference)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        epoch_list.append(epoch+1)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        early_stopping(val_loss, model)

        gpu_mem_alloc = torch.cuda.max_memory_allocated(gpu_index) / 1048576 if enable_cuda and torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.8f} |Train loss: {:.4f} | Val loss: {:.4f} | GPU occupy: {:.2f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))
        print('Training Time:', round(end_training-start_training, 2),'s/epoch')
        print('Inference Time:',round(end_inference-start_inference, 2),'s')
        
        if (epoch+1) % 20 == 0:
            test(model_save_path, zscore, loss, model, test_iter, n_pred, device, False)
        
        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')
    print('Average training Time:', round(np.mean(Training_time_list), 2), 's/epoch')
    print('Average inference Time:', round( np.mean(Inference_time_list),2 ), 's')
    loss_list = pd.DataFrame(np.round(np.array([epoch_list,train_loss_list,val_loss_list]).T, 4),columns=('epoch','tran_loss','val_loss'))
    loss_list.set_index(["epoch"], inplace=True)
    return loss_list

def val(model, val_iter, n_pred, device):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:

            y_pred = model(x.to(device)) 
            l = loss(y_pred, y.to(device))

            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def test(model_save_path, zscore, loss, model, test_iter, n_pred, device, out):
    best_model = model
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = utility.evaluate_model(best_model, loss, test_iter, device)
    print('Test loss {:.6f}'.format(test_MSE))
    
    accuracy_pd, out_unnormalized, target_unnormalized = utility.evaluate_metric(best_model, test_iter, zscore, n_pred, device)
    if out:
        return accuracy_pd, out_unnormalized, target_unnormalized

def data_save(time_index_test, out_unnormalized, target_unnormalized, save_data_path, n_pred = 12):
  
    num = out_unnormalized.shape[1]
    time_list = []
    
    for i in range(n_pred):
        head = i
        tail = i + num
        time_list.append(time_index_test[head:tail])
        pd.DataFrame(out_unnormalized[i], index = time_index_test[head:tail]).to_csv(save_data_path+'out_unnormalized_h'+str(i+1)+'.csv')
        pd.DataFrame(target_unnormalized[i], index = time_index_test[head:tail]).to_csv(save_data_path+'target_unnormalized_h'+str(i+1)+'.csv')
        
    

if __name__ == "__main__":

    SEED = 2024
    set_seed(SEED)


    worker_init_fn(SEED)

    dataset = 'PEMS08'
    save_path = './FDGT_master_0/save/'+dataset+'/'
    gpu_index = 0
    order = 2
    batch_size = 32
    epochs = 200
    train_patience = 30
    gated_act_func = 'glu'    
    Kt = 3

    local_graph = 'ChebNet'
    Ks = 3
    if (local_graph == 'GCN') and (Ks != 2):
        Ks = 2
    local_mat_type = 'sym'  

    future_guided = True
    
    future_opt = False
    future_model = 'guided-' + str(future_guided) + ',' + 'opt-' + str(future_opt)
    mat_type = local_graph +'_' + local_mat_type           # 

    time_mask = datetime.datetime.now().strftime('%m-%d-%H-%M')


    device,  n_his, n_pred, day_slot, time_intvl, time_len, time_start, data, adj_mat, stblock_num, n_feature, n_vertex, drop_rate, opt, learning_rate, weight_decay_rate, step_size, gamma, enable_cuda = get_parameters(dataset, gpu_index)
    print(device)

    zscore, train_iter, val_iter, test_iter, data, time_index_test = data_preparate(future_guided, future_opt, data, device, n_his, n_pred, day_slot, time_len, time_start, batch_size)


    time_pred = n_pred * time_intvl #
    time_pred_str = str(time_pred) + '_mins'


    model_name = 'FDT' + '_' + 'future' + '{' +  future_model + '}' + 'order'+ str(order) + '_batch_' + str(batch_size)
    sys.stdout = Logger('./FDGT_master_0/save/'+dataset+'/'+'log/'+model_name+ '_'+dataset+'_'+str(time_mask)+'.txt')
    

    save_data_path = save_path + 'data/'
    model_save_path = save_path + model_name + '_' + dataset + '_' + time_pred_str + '.pth' #
    print('Time of model execution:',time_mask)
    print('Device is GPU:', gpu_index)
    print('Order is:', order)

    model, loss, early_stopping, optimizer, scheduler = model_main(future_guided, local_graph, adj_mat, mat_type, order, Kt, Ks, batch_size, stblock_num, n_feature, n_his, n_pred, n_vertex, gated_act_func, drop_rate, device, learning_rate, weight_decay_rate, model_save_path, step_size, train_patience,  gamma, opt)


    # Training
    loss_list = train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter, enable_cuda, n_pred, device)

    # Testing
    accuracy_pd, out_unnormalized, target_unnormalized = test(model_save_path, zscore, loss, model, test_iter, n_pred, device, True)
    print('Data saving .......')
    data_save(time_index_test, out_unnormalized, target_unnormalized, save_data_path, n_pred)
    loss_list.to_csv( './FDGT_master_0/save/'+dataset+'/'+'loss/'+model_name+ '_'+dataset+'_'+str(time_mask)+'.csv')
    accuracy_pd.to_csv( './FDGT_master_0/save/'+dataset+'/'+'accuracy/'+model_name+ '_'+dataset+'.csv')
    print('Data is saved.')
    print('The experiment has been completed.')
    stdout_original = sys.stdout
    try:
        sys.stdout = stdout_original
    finally:
        sys.stdout = stdout_original
