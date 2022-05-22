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

from model import RuleEncoder, DataEncoder, Net, NaiveModel, SharedNet, DataonlyNet,lstm_model,bilstm,LSTM_test,BiLSTM_layer


model_info = {'dataonly': {'rule': 0.0},
              'deepctrl': {'beta': [0.1], 'scale': 0.01},
             }

def main():
  parser = ArgumentParser()
  # train/test hyper parameters
  parser.add_argument('--datapath1', type=str, default='data/coordinatebTOKYO1new1201.xls')
  parser.add_argument('--datapath2', type=str, default='data/coordinatebTOKYO2new1202.xls')
  parser.add_argument('--datapath3', type=str, default='data/coordinatebTOKYO3d.xls')
  parser.add_argument('--datapath4', type=str, default='data/coordinatebHSBC11d.xls')
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--batch_size', type=int, default=1, help='default: 64')
  parser.add_argument('--model_type', type=str, default='dataonly')
  parser.add_argument('--seed', type=int, default=5) #42
  parser.add_argument('--input_dim_encoder', type=int, default=16)
  parser.add_argument('--output_dim_encoder', type=int, default=16)
  parser.add_argument('--hidden_dim_encoder', type=int, default=128)#512
  parser.add_argument('--hidden_dim_db', type=int, default=128)#512
  parser.add_argument('--n_layers', type=int, default=1)
  parser.add_argument('--epochnum', type=int, default=1000, help='default: 1000')
  parser.add_argument('--epochs', type=int, default=100, help='default: 1000')
  parser.add_argument('--early_stopping_thld', type=int, default=10, help='default: 10')
  parser.add_argument('--valid_freq', type=int, default=5, help='default: 5')


  args = parser.parse_args()
  print(args)
  print()

  device = args.device
  seed = args.seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  datapath1 = args.datapath1
  datapath2 = args.datapath2
  datapath3 = args.datapath3
  datapath4 = args.datapath4
  # Load dataset
  '''
  data1 = pd.read_excel(datapath1,'cordination', usecols=["rx1下面的人","rx2下面的人","by1下面的人","by2下面的人","rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data2 = pd.read_excel(datapath2,'cordination', usecols=["rx1下面的人","rx2下面的人","by1下面的人","by2下面的人","rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data1 = data1.dropna()
  data2 = data2.dropna()
  '''
  data1_new = pd.read_excel(datapath1,'cordination', usecols=["rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data2_new = pd.read_excel(datapath2,'cordination', usecols=["rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data3_new = pd.read_excel(datapath3,'cordination', usecols=["rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data4_new = pd.read_excel(datapath4,'cordination', usecols=["rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data1_new = data1_new.dropna()
  data3_new = data3_new.dropna()
  data2_new = data2_new.dropna()
  data4_new = data4_new.dropna()

  label1=pd.read_excel(datapath1,'cordination', usecols=["d"])
  label2=pd.read_excel(datapath2,'cordination', usecols=["d"])
  label3=pd.read_excel(datapath3,'cordination', usecols=["d"])
  label4=pd.read_excel(datapath4,'cordination', usecols=["d"])
  label1 = label1.dropna()
  label2 = label2.dropna()
  label3 = label3.dropna()
  label4 = label4.dropna()
  #上下半場合併
  
  input_data=pd.concat([data1_new, data2_new, data3_new], ignore_index=True)
  label_data=pd.concat([label1, label2, label3], ignore_index=True)
  test_data=data4_new
  label_test=label4
  
  #上下半場未合併
  '''
  input_data=data1_new
  label_data=label1
  test_data=data2_new
  label_test=label2
  '''
  #normalization
  
  column_trans = ColumnTransformer(
      [('rx1up_norm', StandardScaler(), ['rx1上面的人']),
       ('rx2up_norm', StandardScaler(), ['rx2上面的人']),
       ('by1up_norm', StandardScaler(), ['by1上面的人']),
       ('by2up_morm', StandardScaler(), ['by2上面的人']),
      ], remainder='passthrough'
  )
  '''
  column_trans = ColumnTransformer(
      [('rx1down_norm', StandardScaler(), ['rx1下面的人']),
       ('rx2down_norm', StandardScaler(), ['rx2下面的人']),
       ('by1down_norm', StandardScaler(), ['by1下面的人']),      
       ('by2down_norm', StandardScaler(), ['by2下面的人']),
       ('rx1up_norm', StandardScaler(), ['rx1上面的人']),
       ('rx2up_norm', StandardScaler(), ['rx2上面的人']),
       ('by1up_norm', StandardScaler(), ['by1上面的人']),
       ('by2up_morm', StandardScaler(), ['by2上面的人']),
      ], remainder='passthrough'
  )
  '''
  column_trans2 = ColumnTransformer(
    [('d_norm', OneHotEncoder(), ['d']),
      ], remainder='passthrough'
  )
  
  input_data_trans = column_trans.fit_transform(input_data)
  label_data_trans = column_trans2.fit_transform(label_data)
  test_data_trans = column_trans.fit_transform(test_data)
  label_test_trans = column_trans2.fit_transform(label_test)
  
  print(input_data_trans.shape)
  num_samples = input_data_trans.shape[0]
  input_data_np = input_data_trans.copy()
  print("data read success!")
  

  #create dataset
  look_back = 5
  X_train= create(input_data_trans, look_back)
  X_test= create(test_data_trans, look_back)
  #X_test= create(X_test, look_back)
  #X_train=X_train.reshape(X_train.shape[0], 32, 1)
  print("X_train.shape: "+str(X_train.shape))
  
  
  Y_train = label_data_trans[2:-2]
  Y_test = label_test_trans[2:-2]
  print("Y_train.shape: "+str(Y_train.shape))
  print("X_train.shape: "+str(X_train.shape))
  X_train, Y_train = torch.tensor(X_train, dtype=torch.float32, device=device), torch.tensor(Y_train, dtype=torch.float32, device=device)
  X_test, Y_test = torch.tensor(X_test, dtype=torch.float32, device=device), torch.tensor(Y_test, dtype=torch.float32, device=device)
  batch_size = args.batch_size
  print("train_X.shape: "+str(X_train.shape))
  print("train_y.shape: "+str(Y_train.shape))
  train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
  print("data size: {}".format(len(X_train)))


  model_type = args.model_type
  #add rule
  if model_type=='deepctrl' :
      print('model_type: {}'.format(model_type))
      model_params = model_info[model_type]
      lr = model_params['lr'] if 'lr' in model_params else 0.001
      scale = model_params['scale'] if 'scale' in model_params else 1.0
      beta_param = model_params['beta'] if 'beta' in model_params else [1.0]

      if len(beta_param) == 1:
        alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
      elif len(beta_param) == 2:
        alpha_distribution = Beta(float(beta_param[0]), float(beta_param[1]))


      merge = 'cat'
      input_dim =4 
      output_dim_encoder = args.output_dim_encoder
      hidden_dim_encoder = args.hidden_dim_encoder
      hidden_dim_db = args.hidden_dim_db
      output_dim = 3
      n_layers = args.n_layers
      
      rule_encoder = RuleEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
      data_encoder = DataEncoder(input_dim, output_dim_encoder, hidden_dim_encoder)
      model = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge).to(device) 
      print(model)
      optimizer = optim.Adam(model.parameters(), lr=lr)        
      #loss_rule_func = lambda x,y,z: torch.mean(F.relu(x-z)+F.relu(y-z))  #let x,y do not>z 
      #loss_rule = loss_rule_func(p1, p2, score)
      loss_rule_func = lambda x,y,a,b: torch.mean(F.relu(abs((x-y))+F.relu(abs(a-b)))) 
      #loss_rule_func = lambda x,y,a,b: torch.mean(F.relu(x-y)+F.relu(a-b)) 
      loss_task_func=nn.CrossEntropyLoss()
  

      epochs = args.epochs
      early_stopping_thld = args.early_stopping_thld
      counter_early_stopping = 1
      valid_freq = args.valid_freq  
      best_val_loss = float('inf')
      loss_list=[]
      acc_list=[]
      task_loss_count=0
      n_list=[]
      rule_loss_list=[]
      task_loss_list=[]
      n=0 
      train_ratio=0
      
      model_name='model_name'
      saved_filename = '{}.pt'.format(model_name)
      saved_filename =  os.path.join('file_name', saved_filename)
 
      model.train()
      for epoch in range(1, epochs+1): 
      #for epoch in range(1, 2):       
        total_acc_arr=[]   
        pred_p1=0
        pred_p2=0
        #pred_score=21
        true_p1=0
        true_p2=0
        #true_score=21
        pred_cur=0
        true_cur=0
        c_pp1=0
        c_pp2=0
        c_tp1=0
        c_tp2=0
        #c_tp1=60
        #c_tp2=60
      
       
        for batch_train_x, batch_train_y in train_loader:
<<<<<<< HEAD
=======

>>>>>>> 52baf7390538d774470b3c7390191f2e228c2cee
          if model_type.startswith('ruleonly'):
            alpha = 1.0
          elif model_type.startswith('deepctrl'):
            alpha = alpha_distribution.sample().item()

        
          optimizer.zero_grad()
          pred = model(batch_train_x, alpha=alpha)
          pred=pred.squeeze(1)    
          arg_pred=torch.argmax(pred,axis=1)
          arg_label=torch.argmax(batch_train_y,axis=1)
          acc_arr=torch.eq(arg_label,arg_pred).numpy().tolist()
          total_acc_arr+=acc_arr
          pred_int=arg_pred.item()        
          true_int=arg_label.item() 
          
         
          #count pred p1,p2 
          if pred_cur==0:
            if pred_int==1:
              #pred_p1+=1
              c_pp1+=1
              #if pred_p1==pred_p2 and pred_p1>21:
               #   pred_score+=1
            elif pred_int==2:
              #pred_p2+=1
              c_pp2+=1
          else:
            pass
          
          if true_cur==0:
            if true_int==1:
              #pred_p1+=1
              c_tp1+=1
              #if pred_p1==pred_p2 and pred_p1>21:
               #   pred_score+=1
            elif true_int==2:
              #pred_p2+=1
              c_tp2+=1
          else:
            pass
            
          c_tp1=torch.tensor(c_tp1).float()
          c_tp2=torch.tensor(c_tp2).float()
          c_pp1=torch.tensor(c_pp1).float()
          c_pp2=torch.tensor(c_pp2).float()

          loss_task = loss_task_func(pred, batch_train_y)
          #loss_rule = loss_rule_func(c_pp1, c_tp1, c_pp2, c_tp2)
          loss_rule = loss_rule_func(c_tp1, c_pp1, c_tp2, c_pp2)
          task_loss_count+=1
             
          loss = alpha * loss_rule + (1 - alpha) * loss_task
          loss.backward()
          optimizer.step()
          pred_cur=pred_int
          true_cur=true_int
        
        
        acc=sum(total_acc_arr)/len(total_acc_arr)
        acc_list.append(acc)
<<<<<<< HEAD
        loss2=loss.item()
        loss_list.append(loss2)
        loss_task2=loss_task.item()
=======
        loss2=loss.item()  
        loss_list.append(loss2)
        loss_task2=loss_task.item() 
>>>>>>> 52baf7390538d774470b3c7390191f2e228c2cee
        task_loss_list.append(loss_task2)
        loss_rule2=loss_rule.item()
        rule_loss_list.append(loss_rule2)
        n+=1
        n_list.append(n)
      
        if epoch%5==0:
          print('Epoch: {} Loss: {:.6f} acc: {}'.format(epoch, loss,acc))
<<<<<<< HEAD
=======
      
              
>>>>>>> 52baf7390538d774470b3c7390191f2e228c2cee
          torch.save(model,saved_filename)

      print("training success")    
      
      x1 = range(1,101) 
      y1= loss_list
      x2 = range(1,101) 
      y2= acc_list
      x3 = n_list
      y3 = task_loss_list
      x4 = n_list
      y4 = rule_loss_list
      
      
      plt.plot(x2, y2, 'o-')
      plt.title('train_acc')
      plt.ylabel('')
      plt.show()

      plt.plot(x1, y1, 'o-')
      plt.title('train_loss')
      plt.ylabel('')
      plt.show()

      plt.subplot(5, 1, 1)
      plt.plot(x3, y3, 'o-')
      plt.title('task_train_loss')
      plt.ylabel('')
        
        
      plt.subplot(5, 1, 3)
      plt.plot(x4, y4, 'o-')
      plt.xlabel('')
      plt.ylabel('')
      plt.title('rule_train_loss')
      plt.show()

      #test
      test_acc_list=[]
      model_eval=torch.load(saved_filename)

      model_eval.eval()
      test_total_acc_arr=[]
      with torch.no_grad():
        for te_x, te_y in test_loader:
          te_y=te_y.squeeze(1)
          te_x=te_x.squeeze(1)

          output = model_eval(te_x, alpha=1.0)
          test_loss_task = loss_task_func(output, te_y).item()
          
          test_arg_pred=torch.argmax(output,axis=1)
          test_arg_label=torch.argmax(te_y,axis=1)
          test_acc_arr=torch.eq(test_arg_label,test_arg_pred).numpy().tolist()
          test_total_acc_arr+=test_acc_arr
          test_acc=sum(test_total_acc_arr)/len(test_total_acc_arr)

      print('\n[Test] Average loss: {:.8f}  test_acc: {} \n'.format(test_loss_task, test_acc))
      print("len(test_total_acc_arr): "+str(len(test_total_acc_arr)))
      
      test_arg_label_arr=[]
      test_arg_pred_arr=[]
      model_eval.eval()
      alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      
      for alpha in alphas:
        model_eval.eval()
        test_total_acc_arr=[]
        with torch.no_grad():
          for te_x, te_y in test_loader:
            te_y=te_y.squeeze(1)
            te_x=te_x.squeeze(1)
  
            if model_type.startswith('deepctrl'):
              output = model_eval(te_x, alpha=alpha)
            elif model_type.startswith('ruleonly'):
              output = model_eval(te_x, alpha=1.0)

            test_loss_task = loss_task_func(output, te_y).item()

            test_loss_task = loss_task_func(output, te_y).item()
            test_arg_pred=torch.argmax(output,axis=1)
            test_arg_label=torch.argmax(te_y,axis=1)
            test_arg_pred_arr+=test_arg_pred
            test_arg_label_arr+=test_arg_label
            df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])
            
            #save to excel
            excel_name=alpha

            test_acc_arr=torch.eq(test_arg_label,test_arg_pred).numpy().tolist()
            test_total_acc_arr+=test_acc_arr
            test_acc=sum(test_total_acc_arr)/len(test_total_acc_arr)

        print('\n[Test] Average loss: {:.8f}  test_acc: {} (alpha:{})\n'.format(test_loss_task, test_acc,alpha))
        test_arg_pred_arr+=test_arg_pred
        test_arg_label_arr+=test_arg_label
        df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])
            
        #save to excel
   
        excel_name=alpha
        df1.to_excel('file_name/'+str(excel_name)+'.xlsx',index=False)
           
        print('alpha: '+str(excel_name)+' save_scuccessful!')
        test_arg_label_arr=[]
        test_arg_pred_arr=[]
        
      
      #df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])
      df2=pd.DataFrame(test_arg_label_arr,columns=['true'])
      
      #save to excel
      #df1.to_excel('excel_file/pre0416.xlsx',index=False)
    
      #df2.to_excel('excel_file/true0419.xlsx',index=False)  
      #print('true_data_save_scuccessful!')

if __name__ == '__main__':
  main()
