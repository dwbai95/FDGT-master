import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy




def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Differential(nn.Module):
    def __init__(self, T, order, device):
        super(Differential, self).__init__()
        self.linears = []
        self.order = order
        self.device = device
        self.glu = False
        for i in range(order):
            if self.glu:
                self.linears.append(nn.Linear(T-i, (i+1) * 2)) 
            else:
                self.linears.append(nn.Linear(T-i, i+1)) 
        self.linears = nn.ModuleList(self.linears)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        x_list = []
        x_0 = x
        x_list.append(x_0)

        for i in range(self.order):
            x_1 = x_0[:,:, 1:, :] - x_0[:,:,:-1, :]
            
            x_2 = self.linears[i](x_0.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            if self.glu:
                p = x_2[:,:,:(i+1),:]
                q = x_2[:,:,(i+1):,:]
                x_list.append(torch.cat((p * self.sigmoid(q), x_1), dim = 2))
                
            else:
                x_list.append(torch.cat((self.sigmoid(x_2), x_1), dim = 2))
            x_0 = x_1
        x_out = torch.cat(x_list, dim = 1)

        return x_out



class MultiHead(nn.Module):
    def __init__(self, en_mult_g, Ko, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHead, self).__init__()
        assert d_model % h == 0
        self.en_mult_g = en_mult_g
        self.Ko = Ko
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask=None):
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size, c_in, T, n_vertex = x.shape

        x = x.permute(0,2,3,1)
        query, key = [l(x) for l, x in zip(self.linears, (x, x))]
        query, key = [x.view(-1, T, n_vertex, self.h, self.d_k).transpose(2, 3) for x in (query, key)]
        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.en_mult_g:
            scores = torch.einsum('ijkmn->ijmn', [scores]) / math.sqrt(self.d_k)
   
        else:
            scores = torch.einsum('ijkmn->imn', [scores]) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)

        
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn





class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):

        if self.c_in > self.c_out:
            x_align = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, c_in, timestep, n_vertex = x.shape
            x_align = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x_align = x
        return x_align

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
    
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result
    
    
    
    
class Gated_Cauasl_CNN(nn.Module):


    def __init__(self, Kt, c_in, c_out, n_vertex, act_func, enable_gated_act_func):
        super(Gated_Cauasl_CNN, self).__init__()
       
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.Kt = Kt
        self.act_func = act_func
        self.enable_gated_act_func = enable_gated_act_func
        self.align = Align(c_in, c_out)
        self.en_mult_g = True

        
        
        if self.enable_gated_act_func == True:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=  2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
            
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels= c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
            
        self.linear = nn.Linear(n_vertex, n_vertex)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        
        x_causal_conv = self.causal_conv(x)
       
        
        if self.enable_gated_act_func == True:
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            # Temporal Convolution Layer (GLU)
            if self.act_func == 'glu':
                
                x_glu = torch.mul(x_p + x_in , self.sigmoid(x_q))
                x_tc_out = x_glu

            # Temporal Convolution Layer (GTU)
            elif self.act_func == "gtu":
                # Tanh(x_p + x_in) ⊙ Sigmoid(x_q)
                x_gtu = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
                x_tc_out = x_gtu

            else:
                raise ValueError(f'ERROR: activation function {self.act_func} is not defined.')

        else:

            
            if self.act_func == "linear":
                x_linear = self.linear(x_causal_conv + x_in)
                x_tc_out = x_linear
            
     
            elif self.act_func == "sigmoid":
                x_sigmoid = self.sigmoid(x_causal_conv + x_in)
                x_tc_out = x_sigmoid

         
            elif self.act_func == "tanh":
                x_tanh = self.tanh(x_causal_conv + x_in)
                x_tc_out = x_tanh

          
            elif self.act_func == "softsign":
                x_softsign = self.softsign(x_causal_conv + x_in)
                x_tc_out = x_softsign

           
            elif self.act_func == "relu":
                x_relu = self.relu(x_causal_conv + x_in)
                x_tc_out = x_relu

         
            elif self.act_func == "softplus":
                x_softplus = self.softplus(x_causal_conv + x_in)
                x_tc_out = x_softplus
        
          
            elif self.act_func == "leakyrelu":
                x_leakyrelu = self.leakyrelu(x_causal_conv + x_in)
                x_tc_out = x_leakyrelu

         
            elif self.act_func == "prelu":
                x_prelu = self.prelu(x_causal_conv + x_in)
                x_tc_out = x_prelu

        
            elif self.act_func == "elu":
                x_elu = self.elu(x_causal_conv + x_in)
                x_tc_out = x_elu

            else:
                raise ValueError(f'ERROR: activation function {self.act_func} is not defined.')
        
        return x_tc_out
    
    
    
    
    

class LocalConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, LocalConv_matrix, enable_bias, graph_conv_act_func):
        super(LocalConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.LocalConv_matrix = LocalConv_matrix
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))

        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # For Sigmoid, Tanh or Softsign
        if self.graph_conv_act_func == 'sigmoid' or self.graph_conv_act_func == 'tanh' or self.graph_conv_act_func == 'softsign':
            init.xavier_uniform_(self.weight)

        # For ReLU, Softplus, Leaky ReLU, PReLU, or ELU
        elif self.graph_conv_act_func == 'relu' or self.graph_conv_act_func == 'softplus' or self.graph_conv_act_func == 'leakyrelu' \
            or self.graph_conv_act_func == 'prelu' or self.graph_conv_act_func == 'elu':
            init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        
        batch_size, c_in, T, n_vertex = x.shape
        
        x = x.reshape(n_vertex, -1)
        x_0 = x
        x_1 = torch.mm(self.LocalConv_matrix, x)
        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.mm(2 * self.LocalConv_matrix, x_list[k - 1]) - x_list[k - 2])
        x_tensor = torch.stack(x_list, dim=2)

        x_mul = torch.mm(x_tensor.view(-1, self.Ks * c_in), self.weight.view(self.Ks * c_in, -1)).view(-1, self.c_out)

        if self.bias is not None:
            x_LocalConv = x_mul + self.bias
        else:
            x_LocalConv = x_mul
        
        return x_LocalConv

class GlobalConv(nn.Module):
    def __init__(self, c_in, c_out, Kt, gcnconv_matrix, enable_bias, graph_conv_act_func, n_vertex):
        super(GlobalConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.c_out2 = int(c_out * 2)
        self.dropout = 0.5
        self.gcnconv_matrix = gcnconv_matrix
        self.enable_bias = enable_bias
        self.graph_conv_act_func = graph_conv_act_func
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if enable_bias == True:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()
        
        

        self.align = Align(c_in, 1)
        self.tc1_ln = nn.LayerNorm([n_vertex, c_out])
        self.en_mult_g = True
        self.att = MultiHead(self.en_mult_g, Ko = 3, d_model =self.c_out2 , h=4 )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=0.1)
        
        
    def initialize_parameters(self):
        # For Sigmoid, Tanh or Softsign
        if self.graph_conv_act_func == 'sigmoid' or self.graph_conv_act_func == 'tanh' or self.graph_conv_act_func == 'softsign':
            init.xavier_uniform_(self.weight)

        # For ReLU, Softplus, Leaky ReLU, PReLU, or ELU
        elif self.graph_conv_act_func == 'relu' or self.graph_conv_act_func == 'softplus' or self.graph_conv_act_func == 'leakyrelu' \
            or self.graph_conv_act_func == 'prelu' or self.graph_conv_act_func == 'elu':
            init.kaiming_uniform_(self.weight)

        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)

    def forward(self, x):
        batch_size, c_in, T, n_vertex = x.shape
        
        
        
        matrix = self.att(x)
        

        x_first_mul = torch.mm(x.reshape(-1, c_in), self.weight).reshape(batch_size, self.c_out, T, n_vertex)

        x_second_mul = torch.einsum('ijkm,ikmn->ijkn', [x_first_mul, matrix]).reshape(-1, self.c_out)

        if self.bias is not None:
            x_gcnconv = x_second_mul + self.bias
        else:
            x_gcnconv = x_second_mul
        
        return x_gcnconv

class PDG(nn.Module):
    def __init__(self, Ks, c_in, c_out, graph_conv_type, graph_conv_matrix, graph_conv_act_func, n_vertex):
        super(PDG, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.c_out2 = int(c_out/2)
        self.align = Align(c_in, c_out)
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.graph_conv_act_func = graph_conv_act_func
        self.enable_bias = True
        
        self.kt  = 3
        
        self.GlobalConv = GlobalConv(c_out, self.c_out2, self.kt, graph_conv_matrix, self.enable_bias, graph_conv_act_func, n_vertex)
        self.LocalConv = LocalConv(c_out, self.c_out2, Ks, graph_conv_matrix, self.enable_bias, graph_conv_act_func)
        
        
    def forward(self, x):
        x_gc_in = self.align(x)
        batch_size, c_in, T, n_vertex = x_gc_in.shape

        
        x_gc1= self.GlobalConv(x_gc_in)
        x_gc1 = x_gc1.view(batch_size, self.c_out2, T, n_vertex)

        
        x_gc2 = self.LocalConv(x_gc_in).view(batch_size, self.c_out2, T, n_vertex)
        x_gc = torch.cat((x_gc1,x_gc2), dim = 1)

        x_gc_with_rc = torch.add(x_gc, x_gc_in)
        x_gc_out = x_gc_with_rc
        return x_gc_out
    
    
    
class DSCM(nn.Module):

    
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, gated_act_func, graph_conv_type, graph_conv_matrix, drop_rate):
        super(DSCM, self).__init__()

        self.Kt = Kt
        self.Ks = Ks


        
        self.n_vertex = n_vertex
        self.last_block_channel = last_block_channel
        self.channels = channels
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.graph_conv_type = graph_conv_type
        self.graph_conv_matrix = graph_conv_matrix
        self.graph_conv_act_func = 'relu'
        self.drop_rate = drop_rate

        
        self.tmp_conv1 = Gated_Cauasl_CNN(Kt, last_block_channel, channels[0], n_vertex, gated_act_func, self.enable_gated_act_func)
        self.PDG = PDG(Ks, channels[0], channels[1], graph_conv_type, graph_conv_matrix, self.graph_conv_act_func, n_vertex)
        self.tmp_conv2 = Gated_Cauasl_CNN(Kt, channels[1], channels[2], n_vertex, gated_act_func, self.enable_gated_act_func)
    

        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=drop_rate)

    def forward(self, x):
        
        x_tmp_conv1 = self.tmp_conv1(x)
        x_graph_conv = self.PDG(x_tmp_conv1)
        if self.graph_conv_act_func == 'sigmoid':
            x_act_func = self.sigmoid(x_graph_conv)
        elif self.graph_conv_act_func == 'tanh':
            x_act_func = self.tanh(x_graph_conv)
        elif self.graph_conv_act_func == 'softsign':
            x_act_func = self.softsign(x_graph_conv)
        elif self.graph_conv_act_func == 'relu':
            x_act_func = self.relu(x_graph_conv)
        elif self.graph_conv_act_func == 'softplus':
            x_act_func = self.softplus(x_graph_conv)
        elif self.graph_conv_act_func == 'leakyrelu':
            x_act_func = self.leakyrelu(x_graph_conv)
        elif self.graph_conv_act_func == 'prelu':
            x_act_func = self.prelu(x_graph_conv)
        elif self.graph_conv_act_func == 'elu':
            x_act_func = self.elu(x_graph_conv)
        x_tmp_conv2 = self.tmp_conv2(x_act_func)
        x_tmp_out = x_tmp_conv2
        
        
        x_tc2_ln = self.tc2_ln(x_tmp_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_do = self.do(x_tc2_ln)
        x_st_conv_out = x_do
        return x_st_conv_out 
    
    
    
    
class Encode(nn.Module):

    
    def __init__(self, device, Kt, Ks, order, n_feature, T, n_vertex, last_block_channel, channels, gated_act_func, graph_conv_type, graph_conv_matrix, drop_rate):
        super(Encode, self).__init__()

        self.order = order
        if order > 0:
            self.differential = Differential(T, order, device)
            
        self.DSCM = DSCM(Kt, Ks, n_vertex, last_block_channel, channels, gated_act_func, graph_conv_type, graph_conv_matrix, drop_rate)

    def forward(self, x):
        
        
        if self.order > 0:
            
            x_diff = self.differential(x)
        else:
             x_diff = x
            
        output = self.DSCM(x_diff)
        return output

class Decode(nn.Module):

    def __init__(self, future_guided, Kt, Ko, last_block_channel, channels, end_channel, n_vertex, gated_act_func, drop_rate):
        super(Decode, self).__init__()
        
        self.future_guided = future_guided
        self.Ko = Kt
        self.Ko = Ko
        self.end_channel = end_channel
        self.channels = channels
        self.end_channel = end_channel
        self.n_vertex = n_vertex
        self.gated_act_func = gated_act_func
        self.enable_gated_act_func = True
        self.last_block_channel_3 = int(last_block_channel/3)
        self.drop_rate = drop_rate
        self.tmp_conv0 = Gated_Cauasl_CNN(Kt, last_block_channel, last_block_channel, n_vertex, gated_act_func, self.enable_gated_act_func)
        self.tmp_conv1 = Gated_Cauasl_CNN(Ko, last_block_channel, channels[0], n_vertex, gated_act_func, self.enable_gated_act_func)
        
 
        
        self.fc1 = nn.Linear(channels[0], channels[1])
        self.fc2 = nn.Linear(channels[1], end_channel *3)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.act_func = 'sigmoid'
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=self.drop_rate)

    def forward(self, x, y):
  
        x_tc0 = self.tmp_conv0(x)
        x_tc1 = self.tmp_conv1(x_tc0)
        
   
        
        x_tc1_ln = self.tc1_ln(x_tc1.permute(0, 2, 3, 1))
 
        x_fc1 = self.fc1(x_tc1_ln)
        
        if self.act_func == 'sigmoid':
            x_act_func = self.sigmoid(x_fc1)
        elif self.act_func == 'tanh':
            x_act_func = self.tanh(x_fc1)
        elif self.act_func == 'softsign':
            x_act_func = self.softsign(x_fc1)
        elif self.act_func == 'relu':
            x_act_func = self.relu(x_fc1)
        elif self.act_func == 'softplus':
            x_act_func = self.softplus(x_fc1)
        elif self.act_func == 'leakyrelu':
            x_act_func = self.leakyrelu(x_fc1)
        elif self.act_func == 'prelu':
            x_act_func = self.prelu(x_fc1)
        elif self.act_func == 'elu':
            x_act_func = self.elu(x_fc1)

        x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)

        if self.future_guided is True:
      
            gated = self.sigmoid(x_fc2[:,-self.end_channel:,:,:]).reshape(y.shape)
            x_out = torch.mul(x_fc2[:,:self.end_channel,:,:].reshape(y.shape), 1 - gated) + torch.mul(x_fc2[:,self.end_channel:self.end_channel+self.end_channel ,:,:].reshape(y.shape), gated)  + torch.mul(y , gated)
        else:
            x_out = x_fc2
        return x_out