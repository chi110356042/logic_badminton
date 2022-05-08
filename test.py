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
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--batch_size', type=int, default=1, help='default: 64')
  parser.add_argument('--model_type', type=str, default='dataonly')
  parser.add_argument('--seed', type=int, default=5) #42
  parser.add_argument('--input_dim_encoder', type=int, default=16)
  parser.add_argument('--output_dim_encoder', type=int, default=16)
  parser.add_argument('--hidden_dim_encoder', type=int, default=512)
  parser.add_argument('--hidden_dim_db', type=int, default=512)
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

  # Load dataset
  data1 = pd.read_excel(datapath1,'cordination', usecols=["rx1下面的人","rx2下面的人","by1下面的人","by2下面的人","rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data2 = pd.read_excel(datapath2,'cordination', usecols=["rx1下面的人","rx2下面的人","by1下面的人","by2下面的人","rx1上面的人","rx2上面的人","by1上面的人","by2上面的人"])
  data1 = data1.dropna()
  data2 = data2.dropna()

  label1=pd.read_excel(datapath1,'cordination', usecols=["d"])
  label2=pd.read_excel(datapath2,'cordination', usecols=["d"])
  label1 = label1.dropna()
  label2 = label2.dropna()
  #上下半場合併
  '''
  input_data=pd.concat([data1, data2], ignore_index=True)
  label_data=pd.concat([label1, label2], ignore_index=True)
  '''
  #上下半場未合併
  input_data=data1
  label_data=label1
  test_data=data2
  label_test=label2
  
  #normalization
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
  look_back = 4
  X_train= create(input_data_trans, look_back)
  X_test= create(test_data_trans, look_back)
  #X_test= create(X_test, look_back)
  #X_train=X_train.reshape(X_train.shape[0], 32, 1)
  print("X_train.shape: "+str(X_train.shape))
  
  
  Y_train = label_data_trans[3:]
  Y_test = label_test_trans[3:]
  print("Y_train.shape: "+str(Y_train.shape))
  print("X_train.reshape: "+str(X_train.shape))
  X_train, Y_train = torch.tensor(X_train, dtype=torch.float32, device=device), torch.tensor(Y_train, dtype=torch.float32, device=device)
  X_test, Y_test = torch.tensor(X_test, dtype=torch.float32, device=device), torch.tensor(Y_test, dtype=torch.float32, device=device)
  batch_size = args.batch_size
  print("train_X.shape: "+str(X_train.shape))
  print("train_y.shape: "+str(Y_train.shape))
  train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
  print("data size: {}".format(len(X_train)))

  #start
  model_type = args.model_type
  if model_type=='dataonly':
    print('model_type: {}'.format(model_type))
    input_size=8
    hidden_size=128
    output_size=3
    num_layers=1
    dropout=0.3
    bidirectional=True
    #model=bilstm(input_size)
    model=lstm_model(input_size,hidden_size,output_size,num_layers,dropout)
    print(model)
    #criterion=nn.MSELoss()
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.005)
    model_name='no_rule1'
    saved_filename = '{}.pt'.format(model_name)
    saved_filename =  os.path.join('no_rule_model_file', saved_filename)

    
    avg_acc_arr=[]          
    preds=[]
    labels=[]
    loss_list=[]
    acc_list=[]
    model.train()
    for i in range(100):
        total_acc_arr=[]
    
        #total_loss=0
        for idx,(data,label) in enumerate(train_loader):
            #print("data.shape:"+str(data.shape))
            #print("data:"+str(data))
            #print("label:"+str(label))
            label=label.squeeze(1)
            data1=data.squeeze(1) #shape(32,4,8)   (batch_size, seq_len, input_size)
            pred=model(data1)
            pred=pred.squeeze(1) 
  
            label=Variable(label)
            #print("pred:"+str(pred))
            #print("label:"+str(label))
            #print("pred.shape: "+str(pred.shape)) 
            arg_pred=torch.argmax(pred,axis=1)
            #print("label:"+str(label))
            #print("label.shape:"+str(label.shape))
            arg_label=torch.argmax(label,axis=1)
            acc_arr=torch.eq(arg_label,arg_pred).numpy().tolist()
            total_acc_arr+=acc_arr
            
            optimizer.zero_grad()   
            loss = criterion(pred, label)   
            #loss = criterion(pred, label)  
            loss.backward()               
            optimizer.step()            
        
        acc=sum(total_acc_arr)/len(total_acc_arr)
       
        #loss2=loss.detach().numpy()
        loss2=loss.item()
        loss_list.append(loss2)
        acc_list.append(acc)
       
        if i%10==0:
          print('Epoch: {} Loss: {:.6f} acc: {}'.format(i, loss,acc))
          #print(arg_label) 
          #print(arg_pred)
          #print(len(total_acc_arr))
        if i==99:
          torch.save({
              'epoch': i,
              'model_state_dict':model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss
            }, saved_filename)


    print("training success")
    
   

    x1 = range(1,101) 
    y1= loss_list
    x3 = range(1,101) 
    y3= acc_list

    
    plt.plot(x1, y1, 'o-')
    plt.title('train_loss')
    plt.ylabel('')
    plt.show()
   
    plt.plot(x3, y3, 'o-')
    plt.title('train_acc')
    plt.ylabel('')
    plt.show()
    
    #test
    test_acc_list=[]

    model_eval=lstm_model(input_size,hidden_size,output_size,num_layers,dropout)
    checkpoint = torch.load(saved_filename)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    model_eval.eval()
    test_total_acc_arr=[]
    test_arg_pred_arr=[]
    test_arg_label_arr=[]
    with torch.no_grad():
      for te_x, te_y in test_loader:
        #te_y = te_y.unsqueeze(-1)
        te_y=te_y.squeeze(1)
        #print("te_y.shape: "+str(te_y.shape))
        te_x=te_x.squeeze(1)
        #print("te_x.shape: "+str(te_x.shape))

        output = model_eval(te_x)
        test_loss= criterion(output, te_y).item()
        test_arg_pred=torch.argmax(output,axis=1)
        test_arg_label=torch.argmax(te_y,axis=1)

        test_arg_pred_arr+=test_arg_pred
        test_arg_label_arr+=test_arg_label

        test_acc_arr=torch.eq(test_arg_label,test_arg_pred).numpy().tolist()
        test_total_acc_arr+=test_acc_arr
        test_acc=sum(test_total_acc_arr)/len(test_total_acc_arr)

    print('\n[Test] Average loss: {:.8f}  test_acc: {} \n'.format(test_loss, test_acc))
    #print("test_arg_pred: "+str(test_arg_pred))
    #print("test_arg_label: "+str(test_arg_label))
    print("len(test_total_acc_arr): "+str(len(test_total_acc_arr)))
    
    # save result into excel
    df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])
    df2=pd.DataFrame(test_arg_label_arr,columns=['true'])
    df1.to_excel('excel_file/no_rule_predict.xlsx',index=False)
    df2.to_excel('excel_file/no_rule_true.xlsx',index=False)  

  #add rule
  elif model_type=='deepctrl' :
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
      #input_dim =32 #before
      input_dim =8 #after
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
      loss_rule_func = lambda x,y,a,b: torch.mean(F.relu(x-y)+F.relu(a-b)) #let x(c_pp1) do not>y(c_tp1) ,(c_pp2) do not>b(c_tp2)
      #loss_rule = loss_rule_func(c_pp1, c_tp1, c_pp2, c_tp2)
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
      
      model_name='0419'
      saved_filename = 'test_{}.demo.pt'.format(model_name)
      saved_filename =  os.path.join('model_file', saved_filename)
 
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
        c_tp1=19
        c_tp2=20
      
       
        for batch_train_x, batch_train_y in train_loader:

          #batch_train_x=torch.reshape(batch_train_x, (-1, 32))
          #print("batch_train_x:"+str(batch_train_x))
          #print("batch_train_y:"+str(batch_train_y))
          #batch_train_x=batch_train_x.reshape(batch_train_x[0],32)
          #batch_train_y = batch_train_y.unsqueeze(-1)
          if model_type.startswith('ruleonly'):
            alpha = 1.0
          elif model_type.startswith('deepctrl'):
            #alpha = 1.0
            alpha = alpha_distribution.sample().item()

        
          optimizer.zero_grad()
          pred = model(batch_train_x, alpha=alpha)
          #print("pred.shape: "+str(pred.shape))
          pred=pred.squeeze(1)    
          #print("pred.shape: "+str(pred.shape))
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
              #if pred_p1==pred_p2 and pred_p2>21:
               #   pred_score+=1
          
          #count true p1,p2
          '''
          if true_cur==0:
            if true_int==1:
              true_p1+=1
              c_tp1+=1
            elif true_int==2:
              true_p2+=1
              c_tp2+=1
          '''

          # stable output
          #p1=torch.tensor(p1).float()
          #p2=torch.tensor(p2).float()
          #score=torch.tensor(score).float()
    
          c_tp1=torch.tensor(c_tp1).float()
          c_tp2=torch.tensor(c_tp2).float()
          c_pp1=torch.tensor(c_pp1).float()
          c_pp2=torch.tensor(c_pp2).float()

          loss_task = loss_task_func(pred, batch_train_y)
          loss_rule = loss_rule_func(c_pp1, c_tp1, c_pp2, c_tp2)
       
          task_loss_count+=1
         
          #loss_rule = loss_rule_func(p1, p2, score) 
             
          loss = 1 * alpha * loss_rule + (1 - alpha) * loss_task
          loss.backward()
          optimizer.step()
          pred_cur=pred_int
          true_cur=true_int
       
        acc=sum(total_acc_arr)/len(total_acc_arr)
        acc_list.append(acc)
        loss2=loss.item()
        #loss2=loss.detach().numpy()
        loss_list.append(loss2)
        loss_task2=loss_task.item()
        #loss_task2=loss_task.detach().numpy()
        task_loss_list.append(loss_task2)
        #loss_rule2=loss_rule.detach().numpy()
        loss_rule2=loss_rule.item()
        rule_loss_list.append(loss_rule2)
        n+=1
        n_list.append(n)

        #train_ratio += verification(p1, p2, score, threshold=0.0).item()
      
        if epoch%5==0:
          #print("loss_count: "+str(loss_count))
          print('Epoch: {} Loss: {:.6f} acc: {}'.format(epoch, loss,acc))
          print("c_pp1:",c_pp1)
          print("c_pp2:",c_pp2)
          print("c_tp1:",c_tp1)
          print("c_tp2:",c_tp2)
       
          torch.save({
              'epoch': epoch,
              'model_state_dict':model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': best_val_loss
            }, saved_filename)
   

          #print(arg_label) 
          #print(arg_pred)
          #print(len(total_acc_arr))

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
      model_eval = Net(input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge)
      checkpoint = torch.load(saved_filename)
      model_eval.load_state_dict(checkpoint['model_state_dict'])


      model_eval.eval()
      test_total_acc_arr=[]
      with torch.no_grad():
        for te_x, te_y in test_loader:
          #te_x=torch.reshape(te_x, (-1, 32))
          #te_y = te_y.unsqueeze(-1)
          te_y=te_y.squeeze(1)
          #print("te_y.shape: "+str(te_y.shape))
          te_x=te_x.squeeze(1)
          #print("te_x.shape: "+str(te_x.shape))

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
            #te_x=torch.reshape(te_x, (-1, 32))
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
            #if alpha==1.0:
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
        df1.to_excel('excel_file/'+str(excel_name)+'0419.xlsx',index=False)
           
        print('alpha: '+str(excel_name)+' save_scuccessful!')
        test_arg_label_arr=[]
        test_arg_pred_arr=[]
        
      
      #df1=pd.DataFrame(test_arg_pred_arr,columns=['pred'])
      df2=pd.DataFrame(test_arg_label_arr,columns=['true'])
      
      #save to excel
      #df1.to_excel('excel_file/pre0416.xlsx',index=False)
    
      df2.to_excel('excel_file/true0419.xlsx',index=False)  
      print('true_data_save_scuccessful!')

if __name__ == '__main__':
  main()