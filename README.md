# FDGT-master
Future-heuristic Differential Graph Transformer for Traffic Flow Forecasting

This is a PyTorch implementation of FDGT in the following paper: \
Dewei Bai, et al. Future-heuristic Differential Graph Transformer for Traffic Flow Forecasting.



## Setup Python environment for FDGT
Install python environment
```{bash}
$ conda env create -f environment.yml 
```

## Run the Model 

To ensure that the directory is correct, just use the command:

```
python directory_correction.py 
```

To train the model on different datasets just use the command:

```
python Run_FDGT.py 
```

By default it will run the experiments on PEMS4 dataset. 
To select another dataset open run.py and modify DATASET = 'PEMS0X' 
where X is one of the datasets [3,4,7,8]. 

The configurations file are located in the config directory. For changing any of the hyper-parameters modify the conf file 
associated with the dataset and rerun the above command.

