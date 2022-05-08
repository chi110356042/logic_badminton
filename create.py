import numpy as np
import pandas as pd
import torch


def create(dataset, look_back=1): 
    dataX= []
    for i in range(len(dataset)-look_back+1):
       # for j in range(look_back):#i:(i+look_back)
         dataX.append(dataset[i:(i+look_back), 0:8])
    return np.array(dataX)


'''
def verification(p1, p2, score, threshold=score):
  
  return the ratio of qualified samples.

  if isinstance(p1, torch.Tensor):
    return 1.0*torch.sum((p1-score)+(p2-score)< threshold) / curr_E.shape[0]
  else:
    return 1.0*np.sum((p1-score)+(p2-score) < threshold) / curr_E.shape[0]
'''

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
'''

 
x1=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100] 
y1=[2,10,25,20,28,23,26,22,18,23,22,19,19,18,25,19,19,21,19,19]
y2=[78,59,34,34,25,34,28,26,24,23,23,22,24,25,24,25,22,27,20,20]
y3=[19]*20
y4=[20]*20

plt.style.use("ggplot")
plt.plot(x1, y1, ':',label='pred_p1(0 -> 1)')
plt.plot(x1, y2, '.-',label='pred_p2(0 -> 2)')
plt.plot(x1, y3, '-',label='true_p1(0 -> 1)')
plt.plot(x1, y4, '-',label='true_p2(0 -> 2)')
plt.legend(loc='best')
plt.xlabel('epoch') 
plt.xticks(x1)
plt.title('point_change')
plt.show()
'''