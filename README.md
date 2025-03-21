# FDGT-master

This is a PyTorch implementation of FDGT, as described in the following paper:  

> Bai, D., Xia, D., Wu, X., Huang, D., Hu, Y., Tian, Y., ... & Li, H. (2025).  
> Future-heuristic differential graph transformer for traffic flow forecasting.  
> Information Sciences, 701, 121852.  

🚀 **You are welcome to cite our work!**

---

## **📝 Project Description**
FDGT (Future-heuristic Differential Graph Transformer) is a deep learning model designed for **traffic flow forecasting**.  
It integrates **graph neural networks (GNNs) and transformers**, leveraging **future-guided mechanisms** to enhance predictive accuracy.  

### **✨ Key Features**
- **Future-Heuristic Module**: Incorporates future statistical deviations to refine predictions.  
- **Differential Graph Transformer**: Captures spatial-temporal dependencies in traffic networks.  
- **Scalability & Generalization**: Works on multiple traffic datasets (PEMS3, PEMS4, PEMS7, PEMS8).  
- **Configurable Hyperparameters**: Easy tuning via configuration files.

---

## **🔧 Setup Python Environment for FDGT**
To install dependencies, use the following command:
```bash
conda env create -f environment.yml 

## **Run the Model**

- **To ensure that the directory is correct, just use the command: python directory_correction.py


- **To train the model on different datasets just use the command: python Run_FDGT.py
