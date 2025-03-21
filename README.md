# FDGT-master

This is a PyTorch implementation of FDGT, as described in the following paper:  

> Dewei Bai, Dawen Xia, et al.  
> Future-heuristic differential graph transformer for traffic flow forecasting.  
> Information Sciences, 2025, 701, 121852.  

🚀 **Welcome to cite our work!**

---

## **📝 Abstract**
Traffic Flow Forecasting (TFF) is crucial for various Intelligent Transportation System (ITS) applications, including route planning and emergency management. TFF is challenging due to the dynamic spatiotemporal patterns exhibited  by traffic flow. However, existing TFF methods rely on the "average" spatiotemporal patterns for prediction. To this end, this study proposes a heuristic-aware model named "Future-heuristic Differential Graph Transformer" (FDGT) for TFF with dynamic spatiotemporal patterns. Specifically, we define a heuristic knowledge, called "future statistic" which provides reference information to describe the status of an object in the future. Then, we embed these statistics as coding features in the temporal domain of inputs. Next, we utilize Higher-order Differential Neural Networks (HDNNs) to enhance the perception of variation trends in the series. Moreover, we employ a Dual Spatiotemporal Convolutional Module (DSCM) to simultaneously enable the learning of global and local spatiotemporal dependencies. Finally, the \textcolor{red}{Future-heuristic Fusion (FF)} adaptively optimizes the weight distribution of each component, dynamically fuses the decoder's initial prediction and future statistics, and improves the model's generalization ability to capture spatiotemporal heterogeneities at different times. Experimental results on four public datasets demonstrate that FDGT outperforms existing state-of-the-art TFF methods while maintaining superior execution efficiency.  

### **✨ Key Features**
- **Future-Heuristic mechanisms**: Incorporates future statistical deviations to refine predictions.  
- **Scalability & Generalization**: Works on multiple traffic datasets (PEMS3, PEMS4, PEMS7, PEMS8).  
- **Configurable Hyperparameters**: Easy tuning via configuration files.

---

## **🔧 Setup Python Environment for FDGT**
To install dependencies, use the following command:
```bash
conda env create -f environment.yml 
```

## **Run the Model**

- To ensure that the directory is correct, just use the command:
```bash
python directory_correction.py
```

- To train the model on different datasets just use the command:
```bash
python Run_FDGT.py
```
