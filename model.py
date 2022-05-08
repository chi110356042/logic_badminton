import torch 
import torch.nn as nn
import torch.nn.functional as F


class NaiveModel(nn.Module):
  def __init__(self):
    super(NaiveModel, self).__init__()
    self.net = nn.Identity()

  def forward(self, x, alpha=0.0):
    return self.net(x)
'''
class RuleEncoder(nn.Module):
    def __init__(self, input_size):
        super(RuleEncoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 會用全0的 state
        out = self.out(r_out)
        return out

class DataEncoder(nn.Module):
    def __init__(self, input_size):
        super(DataEncoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  
        out = self.out(r_out)
        return out
'''
'''
class RuleEncoder(nn.Module):
  def __init__(self,input_feature_dim,hidden_feature_dim,hidden_layer_num,batch_size,classes_num):
    super(RuleEncoder, self).__init__()
    self.input_feature_dim=input_feature_dim
    self.hidden_feature_dim=hidden_feature_dim
    self.hidden_layer_num=hidden_layer_num
    self.batch_size=batch_size 
         
    #initialize       
    self.lstm=nn.LSTM(input_feature_dim,hidden_feature_dim,hidden_layer_num)
    self.linear1=nn.Linear(hidden_feature_dim,classes_num)

  def forward(self,input):
        h0=torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        c0=torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        output,(hn,cn)=self.lstm(input,(h0,c0))  
        output=self.linear1(output[-1])         
        return output,(hn,cn)

class DataEncoder(nn.Module):
  def __init__(self,input_feature_dim,hidden_feature_dim,hidden_layer_num,batch_size,classes_num):
    super(DataEncoder, self).__init__()
    self.input_feature_dim=input_feature_dim
    self.hidden_feature_dim=hidden_feature_dim
    self.hidden_layer_num=hidden_layer_num
    self.batch_size=batch_size 
         
    #initialize       
    self.lstm=nn.LSTM(input_feature_dim,hidden_feature_dim,hidden_layer_num)
    self.linear1=nn.Linear(hidden_feature_dim,classes_num)

  def forward(self,input):
        h0=torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        c0=torch.randn(self.hidden_layer_num,self.batch_size,self.hidden_feature_dim)
        output,(hn,cn)=self.lstm(input,(h0,c0))  
        output=self.linear1(output[-1])         
        return output,(hn,cn)
'''
class BiLSTM_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first=False):
        super(BiLSTM_layer, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )

        self.fc = nn.Linear(hidden_size, 3)
        

    def forward(self, inputs):
        out, (h_n, c_n) = self.lstm(inputs, None)
        outputs = self.fc(torch.mean(h_n.squeeze(0), dim=0))

        return outputs


class DataEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=3):
    super(DataEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim)
                            )
  def forward(self, x):
    return self.net(x)

class RuleEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=3):
    super(RuleEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim)
                            )

  def forward(self, x):
    return self.net(x)


class DataonlyNet(nn.Module):
  def __init__(self, input_dim, output_dim, data_encoder, hidden_dim=3, n_layers=2, skip=False, input_type='state'):
    super(DataonlyNet, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    self.input_dim_decision_block = self.data_encoder.output_dim

    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    data_z = self.data_encoder(x)

    return data_z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    data_z = self.data_encoder(x)
    z = data_z

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values


class Net(nn.Module):
  def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=4, n_layers=2, merge='cat', skip=False, input_type='state'):
    super(Net, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    assert self.rule_encoder.input_dim ==  self.data_encoder.input_dim
    assert self.rule_encoder.output_dim ==  self.data_encoder.output_dim
    self.merge = merge
    if merge == 'cat':
      self.input_dim_decision_block = self.rule_encoder.output_dim * 2
    elif merge == 'add':
      self.input_dim_decision_block = self.rule_encoder.output_dim

    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      #self.net.append(nn.Linear(in_dim, out_dim))
      self.net.append(BiLSTM_layer(
              input_size=in_dim,
              hidden_size=256,
              num_layers=1,
              bidirectional=True,
              batch_first=True
          ))

      '''
      if i != n_layers-1:
        self.net.append(nn.ReLU())
      '''
    #self.net.append(nn.Sigmoid())
    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    return z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values


class SharedNet(nn.Module):
  def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=4, n_layers=2, merge='cat', skip=False, input_type='state'):
    super(SharedNet, self).__init__()
    self.skip = skip
    self.input_type = input_type
    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.n_layers = n_layers
    assert self.rule_encoder.input_dim ==  self.data_encoder.input_dim
    assert self.rule_encoder.output_dim ==  self.data_encoder.output_dim
    self.merge = merge
    if merge == 'cat':
      self.input_dim_decision_block = self.rule_encoder.output_dim * 2
    elif merge == 'add':
      self.input_dim_decision_block = self.rule_encoder.output_dim
    self.shared_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, self.rule_encoder.input_dim))
    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    out = self.shared_net(x)

    rule_z = self.rule_encoder(out)
    data_z = self.data_encoder(out)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    return z

  def forward(self, x, alpha=0.0):
    # merge: cat or add
    out = self.shared_net(x)

    rule_z = self.rule_encoder(out)
    data_z = self.data_encoder(out)

    if self.merge=='add':
      z = alpha*rule_z + (1-alpha)*data_z    # merge: Add
    elif self.merge=='cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)    # merge: Concat
    elif self.merge=='equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)    # merge: Concat

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:,-1,:]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values

class lstm_model(nn.Module): 
    def __init__(self,input_size,hidden_size,output_size,num_layers,dropout):
        super(lstm_model, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.dropout=dropout
        #self.dropout=dropout
        self.rnn=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,dropout=self.dropout)
        #self.rnn=nn.Sequential(
          #nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True))
        self.linear=nn.Linear(self.hidden_size,self.output_size)
        #self.linear=nn.Sequential(
          #nn.Linear(self.hidden_size,self.output_size),F.softmax(True))
     

 
    def forward(self,x):     
        out,(hidden,cell)=self.rnn(x) # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a,b,c=hidden.shape
        #out=self.linear(hidden.squeeze(0))
        #out = self.linear(out[:, -1, :])
        out=self.linear(hidden.reshape(a*b,c))
        out_fc_relu=F.relu(out)
        #return out
        return out_fc_relu
        


class LSTM_test(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes, num_layers,bidirectional,dropout):
        super(LSTM_test, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout=dropout
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,batch_first=True,num_layers=self.num_layers,bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size*2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        #batch_size, seq_len = x.shape
        batch_size=32
        #初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
        #维度[layers, batch, hidden_len]
        if self.bidirectional:
            h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size)
            c0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        #x = self.lstm(x)
        out,(_,_)= self.lstm(x)
        
        output = self.fc(out[:,-1,:]).squeeze(0) #因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output

class bilstm(nn.Module):
  def __init__(self,input_size):
    super(bilstm,self).__init__()
    self.rnn = nn.LSTM(
      input_size=input_size,
      hidden_size=64,
      num_layers=1,
      batch_first=True,
      bidirectional=True,
      dropout=0.4
    )
    self.out=nn.Sequential(
      nn.Linear(128,1)
    )
  
  def forward(self,x):
    r_out,(h_n,c_n)=self.rnn(x,None)
    out=self.out(r_out[:-1])
    return out