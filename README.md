# TA-RNN
In this study, we propose two deep learning architectures based on Recurrent Neural Networks (RNN), namely Time-aware Attention-based RNN (TA-RNN) and TA-RNN-Autoencoder (TA-RNN-AE) for early predicting of clinical outcomes in Electronic Health Record (EHR) at the next visit and multiple visits ahead for patients, respectively. To mitigate the impact of the irregular time intervals between visits, we propose incorporating time embedding of the elapsed times between consecutive visits. For results interpretability, we propose employing a dual-level attention mechanism that operates between visits and features within each visit.

# Time-aware Attention-based Recurrent Neural Networks (TA-RNN)

![TA-RNN](https://github.com/yourusername/yourrepository/blob/main/path/to/your/image.jpg)


TA-RNN is a deep learning architecture that comprises of three fundamental parts, namely, time embedding, attention-based RNN, and multi-layer perceptron (MLP). TA-RNN is designed for early predicting of clinical outcome in the EHR at the next visit for patients.

# Time-aware Attention-based Recurrent Neural Networks AutoEncoder (TA-RNN-AE)

![TA-RNN-AE](https://github.com/yourusername/yourrepository/blob/main/path/to/your/image.jpg)


TA-RNN-AE is a deep learning architecture that comprises of three fundamental parts, namely, time embedding, attention-based RNN autoencoder, and MLP. TA-RNN-AE is designed for early predicting of clinical outcome in the EHR at multiple visits ahead for patients.

# Parameter learning and evaluation metrics

To increase the prediction’s sensitivity for both architectures, all trainable parameters for the RNN, RNN autoencoder, and MLP were learned in an integral way using a customized binary cross-entropy loss function to give more weight on predicting future clinical outcome in the EHR, seeking to minimize the false negative cases while predicting future clinical outcome of the visit which leads to increased sensitivity of the predictive model.
  
RNN cell, number of epochs, batch size, dropout rate, L2 regularization, and hidden size are the hyperparameters that have been tuned. For model evaluation, F2 score and sensitivity were used.

# Datasets and input format

We evaluated the proposed architectures using three experimental setups. In the first setup, Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset was utilized to train and test the proposed architectures using the longitudinal multi-modal and the cross-sectional demographic data. In the second setup, the models were trained on the entire ADNI longitudinal and cross-sec-tional data and tested on National Alzheimer’s Coordinating Center (NACC) dataset. In the third setup, the Medical Information Mart for Intensive Care (MIMIC-III) dataset was utilized to train and test the proposed architectures. 
  
 - To get ADNI, you need to request an access for it https://adni.loni.usc.edu/
  
 - To get NACC, you need to request an access for it https://naccdata.org/

 - To get MIMIC-III, you need to request an access for it https://mimic.mit.edu/docs/gettingstarted/
  
For ADNI and NACC dataset, the training and test longitudinal data format is given where data is stored as a list containing 3 dimensionals tensors such as [number of samples , number of visits , number of longitudinal feature in each vist]. The training and test demographic data format is given where data is stored as a list containing other lists such that each inner list represents demographic features for one sample. The training and test label data format is given where data is stored as a list containing 3 dimensionals tensors such as [number of samples , number of visits , 1] where the third dimension can be 0 (MCI) or 1 (Dementia).

For MIMIC-III dataset, We carried out preprocessing using identical procedures as employed in the RETAIN method (https://github.com/mp2893/retain) to extract patients’ visits, mortality labels, and time information.
  
 # How to generate pkl files for ADNI data 
 
pkl_files_preperation.ipynb can be used to generate pkl files from raw files (Sample of raw data) with the following assumptions:
  - You have access to ADNI dataset https://adni.loni.usc.edu/, and you already downloaded ADNI_Merge.csv file.
  - You preprocessed ADNI_Merge.csv file (removing unnecessary columns, taking care of NAN and missing values, removing patients with single visit, and removing patients diagnoses as cognitively normal).
  - ADNI_Merge.csv is split into two files: longitudinal_data.csv and demographic_data.csv.
  - longitudinal_data.csv should have 'RID', 'VISCODEE', 'DX, and at least one longitudinal feature. In this file, each record represents one visit, so same RID can have multiple visits.
  - demographic_data.csv should have 'RID' and at least one demographic feature. In this file, each record represents demographic data for one patient.
  - All files (pkl_files_preperation.ipynb, longitudinal_data.csv, and demographic_data.csv) should in the same directory.
  - Open and run pkl_files_preperation.ipynb using Jupyter Notebook. You will be asked to determine the number of visits that you would like to use to train the model and the number of future visits that you would like to predict thier diagnosis.
  - For TA-RNN, the number of future visits that you would like to predict thier diagnosis is always 1.


Sample of longitudinal_data.csv and demographic_data.csv are provided in Raw data sample folder

After running the the code without any errors, following files will be generated:
 - longitudinal_data_train.pkl

 - label_train.pkl 

 - demographic_data_train.pkl

 - elapsed_data_train.pkl

 - longitudinal_data_test.pkl

 - label_test.pkl

 - demographic_data_test.pkl

 - elapsed_data_test.pkl
 
 # Compitability
 
 All codes are compatible with Tensorflow version 2.14.0, Keras version 2.14.0 and Python 3.11.5.
 
 # How to run TA-RNN
 
 To run TA-RNN, you have to have the following files in the same directory:
 
  - TA-RNN.ipynb
  - longitudinal_data_train.pkl
  - label_train.pkl 
  - demographic_data_train.pkl
  - elapsed_data_train.pkl
  - longitudinal_data_test.pkl
  - label_test.pkl
  - demographic_data_test.pkl
  - elapsed_data_test.pkl
  - hp_df.csv which represents values of hyperparameters that have been tuned
 
After you put all files in the same directory, open and run TA-RNN.ipynb using Jupyter Notebook. TA-RNN will be trained and tested five times and results will be generated as csv file with the following format (x_y_TA-RNN.csv) where x means the number of visits that have been used to train the model and y means the number of future visits for prediction.

To change values of hyperparameters, open hp_df.csv and change values. The values should be as following:
 - batch_size: integer
 - epoch: integer
 - dropout: float number
 - l2: float number 
 - cell: one of these values [GRU, LSTM, biGRU, biLSTM]
 - hidden_s: integer
 - embedding_s: integer
 

# How to run TA-RNN-AE
 
 To runn TA-RNN-AE, you have to have the following files in the same directory:
 
  - TA-RNN-AE.ipynb
  - longitudinal_data_train.pkl
  - label_train.pkl 
  - demographic_data_train.pkl
  - elapsed_data_train.pkl
  - longitudinal_data_test.pkl
  - label_test.pkl
  - demographic_data_test.pkl
  - elapsed_data_test.pkl
  - hp_df.csv which represents values of hyperparameters that have been tuned
 
After you put all files in the same directory, open and run TA-RNN-AE.ipynb using Jupyter Notebook. TA-RNN-AE will be trained and tested five times and results will be generated as csv file with the following format (x_y_TA-RNN-AE.csv) where x means the number of visits that have been used to train the model and y means the number of future visits for prediction.

To change values of hyperparameters, open hp_df.csv and change values. The values should be as following:
 - batch_size: integer
 - epoch: integer
 - dropout: float number
 - l2: float number 
 - cell: one of these values [GRU, LSTM, biGRU, biLSTM]
 - hidden_s: integer
 - embedding_s: integer
