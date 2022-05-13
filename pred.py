from __future__ import division
from __future__ import print_function
from copy import deepcopy
import openpyxl
from openpyxl.workbook import Workbook

import sys
import os
import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta
from create_dataset import *
from test import *

from model import RuleEncoder, DataEncoder, Net, NaiveModel, SharedNet, DataonlyNet,lstm_model,bilstm,LSTM_test,BiLSTM_layer


#load test data
test_data = pd.read_excel('data/coordinatebHSBC11d.xls','cordination', usecols=["rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
test_data_new = test_data.dropna()
input_data=test_data_new

column_trans = ColumnTransformer(
      [('rx1up_norm', StandardScaler(), ['rx1上面的人']),
       ('rx2up_norm', StandardScaler(), ['rx2上面的人']),
       ('by1up_norm', StandardScaler(), ['by1上面的人']),
       ('by2up_morm', StandardScaler(), ['by2上面的人']),
      ], remainder='passthrough'
  )
input_data_trans = column_trans.fit_transform(input_data)
#create dataset
look_back = 5
X_test= create(input_data_trans, look_back)
X_test= torch.tensor(X_test, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(X_test), batch_size=1, shuffle=False)

test_arg_pred_arr=[]
model_name='0513'    #use:best 0513  #new_0509
saved_filename = '{}.pt'.format(model_name)
saved_filename =  os.path.join('model2_file', saved_filename)
#model_eval = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge)
model_eval = torch.load(saved_filename)
print(model_eval)

#model_eval.load_state_dict(checkpoint['model_state_dict'])
with torch.no_grad():
  pred = model_eval(X_test, alpha=0.9) #test1:0.2
  test_arg_pred=torch.argmax(pred,axis=1)
  test_arg_pred_arr+=test_arg_pred
  df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])

#save to excel
   
excel_name='HSBC11_pred_d'
df1.to_excel('pred_file/'+str(excel_name)+'_0513.xlsx',index=False)
           
print('save_scuccessful!')

#calculate acc
pred_data = pd.read_excel('pred2_file/HSBC11_pred_d_0513.xlsx', usecols=["pred"])
true_data = pd.read_excel('data/coordinatebHSBC11d.xls','cordination', usecols=["d"])
print(len(pred_data))
print(len(true_data))
pred_arr=pred_data.values
true_arr=true_data.values
#print(len(pred_arr))
count=0
for i in range(len(pred_arr)):
        if pred_arr[i]==true_arr[i]:
            count+=1
print('acc= ',count/len(pred_arr))

