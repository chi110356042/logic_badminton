import numpy as np
import pandas as pd
import torch


def create(dataset, look_back=1): 
    dataX= []
    for i in range(len(dataset)-look_back+1):
       # for j in range(look_back):#i:(i+look_back)
         dataX.append(dataset[i:(i+look_back), 0:8])
    return np.array(dataX)



#verification_df = pd.read_excel('excel_file/true_data.xlsx',usecols=["true"])
#verification_list = list(verification_df['true'])
verification_df = pd.read_excel('data/coordinatebTOKYO1new1201.xls',usecols=["d"])
verification_list = list(verification_df['d'])
#verification_df = pd.read_excel('excel_file/0.2.xlsx',usecols=["pred"])
#verification_list = list(verification_df['pred'])

change1=0
change2=0
zero=0
one=0
two=0

for i in range(0,len(verification_list)-1):
  if verification_list[i]==0:
    zero+=1
  elif verification_list[i]==1:
    one+=1
  else:
    two+=1

  if verification_list[i] ==0:
    if verification_list[i+1]==1:
      change1+=1
    elif verification_list[i+1]==2:
      change2+=1
print('change1: ',change1)
print('change2: ',change2)
print('zero: ',zero)
print('one: ',one)
print('two: ',two)
