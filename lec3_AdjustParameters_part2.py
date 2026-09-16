#=============================一些概括性的总结========================
#一、这里有哪些地方使用了std？为什么？
#   （1）用 std 来构造参数（初始化）

#   （2）用 std（方差）来归一化激活值

#   （3）用 std 来诊断——评估参数/梯度表现 



#二、有了batch_norm层，是否还需要对线性层参数进行归一化？
#   （1）每一次经过不归一化的线性层，输出的激活值都会更离散。那么多层叠加就会导致激活值过饱和
#       1.如果不归一化W层，那么假设Hin矩阵和W矩阵，初始统计特征是：均值为0，方差为1。大小都为n*n
#       2..假设Hout = Hin@W，那么Hout的统计特征是怎么样的？————均值不变，std变为n**0.5————统计学知识

#   （2）我们设计的神经网络的结构如下：线性层 ---> BN层 ---> tanh层 ---> 下一个线性层
#   （3）分析没有归一化W参数情况下，forward过程，激活值健康状况如何：


#   （4）分析没有归一化W参数情况下，backward过程，激活值健康状况如何：




#=========================cover一些part1的细节还有内容总结========================
# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import scipy.stats as stats
import numpy as np

#===================================init部分====================================
#读入数据集，并且对数据集进行分类
#构造字典，定义上下文长度
words = open('Neural Network_Zero to hero/makemore/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)


block_size = 3 # 也就是上下文长度：读入多少个上文，来预测下一个char
def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%



#===============================此处是线性层的定义=============================
#Weight表示权重矩阵，bias表示偏置矩阵
#fan_in表示上一层的激活值，也就是当前层输入；fan_out表示当前层的激活值，也就是当前层输出
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
#   良好的初始化Weight矩阵，确保数值分布合理
    self.bias = torch.zeros(fan_out) if bias else None
#   一开始不设置任何偏置——————我们希望线性层的输出，大致符合标准正态分布。有了偏置之后，输出的mean就不是0了
  
  def __call__(self, x):
#   __call__是一个魔术方法，允许对象实例被像方法一样调用
#   传入的self是线性层本身，x是上一层的激活值，用做这一层的参数
#   这个方法就是linear层的forward pass方法
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

#   提供外界接口，方便地获取本linear层的参数
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])



#==============================此处是BatchNorm层的定义==========================
class BatchNorm1d:
    
#   初始化方法，生成BN层
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    
    # parameters (trained with backprop)，也就是上一个part的Batch_gain
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    
    # buffers (trained with a running 'momentum update'),也就是上一个part的Batch_bias
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  #接受上一层输入的激励值，由BN层进行归一化
  def __call__(self, x):
    # self是BN层自己，x为上一层传入的激励值
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
      
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    
    #在训练的过程中不断更新逼近全局mean
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]


#===============================激活函数层，采用tanh======================================
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []



#======================================开始训练=======================================
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

#（1）初始化layers，和特殊的嵌入矩阵C
C = torch.randn((vocab_size, n_embd),            generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]


#这段代码作用：训练开始前的"最后一层信心压制"，目的是让网络一开始输出接近均匀分布，而不是"自信地瞎猜"。
with torch.no_grad():
#（2）为什么最后一层的BN层的gamma值，要×一个0.1？
  #输入index是-1，表示选取list的最后一个对象
  #这一层是紧邻着logits的最后一层，把它整体乘 0.1，等于把所有 27 个 logit 的幅度压缩 10 倍。
  #原本的logits的值，应该近似为标准正态分布，那么交叉熵的期望大概是3.80————————看数学原理
  #gamma乘了0.1，相当于把激励值的输出的方差变成了原先的0.01倍，于是交叉熵就变成了大约3.29————具体计算看原理
  #非常接近logits的各分量完全相等时候，对应的交叉熵值
  layers[-1].gamma *= 0.1

#（3）这里目的是，根据每一层的种类，给初始化的参数W们乘以常数gain
#如果没有BN层，我们就希望在每一个tanh层之前，手动对tanh的输入进行归一化，也就是：
#   tanh_input = gain*(tan_input - tanh_input.mean)/tanh_input.std
#   因为tanh是一个压缩函数，比如输入的数方差是1，那么经过tanh之后，输出的数方差大约就是0.63
#   就会导致压缩，所以要提前放大输入的方差，使得tanh输出方差仍然是1左右
#但是这里有了BN层，就可以每层都修正上一层tanh带来的压缩问题，就不需要再使用gain，来手动处理线性层输出，再输入tanh层了
#保留这一段仅仅作为警示
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 1.0 #5/3

#（4）这一段就是为了把所有层的所有参数集中到一起，方便backward反向传播
parameters = [C] + [p for layer in layers for p in layer.parameters()]
#注意，这里的拼接，parameters列表内部的每一个元素是这些参数数组。
#比如parameters[0]是C这个Tensor，而不是C张量里面的第一个元素
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
  
  

# same optimization as last time
max_steps = 200000  #最大训练次数
batch_size = 32     #每一次任意选择Batch_size数量的样本进行训练
lossi = []          #记录，每一轮学习，输出的loss
ud = []             
#ud记录的是

for i in range(max_steps):
  
  #每一步挑选batch_size数量的样本，来训练
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  #前向传播
  emb = C[Xb] # embed the characters into vectors
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # loss function
  
  #backward传播
  for layer in layers:
    layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
#（5）这部分是为了保存非叶子节点tensor内储存的grad，为了调试使用
#       1.我们设定每一层的参数，并不是由前面的层计算而来的，本身就是总的神经网络tree的叶子
#       2.运算产生的中间结果，比如 hpreact = x @ W + b。
#           反向传播时，梯度只是流过它们，算完传给下一层就释放了，不会存进 .grad。
#           所以 hpreact.grad 是 None。
#       3.这里就是告诉Pytorch，去储存每一层layer的输入的激励值的grad，以及输出的激励值的grad
  for p in parameters:
#   在更新梯度之前，清空之前的存在的梯度，再在当次的backward重新计算
#   因为parameter都是叶子tensor，所以默认保存grad。所以每次更新之前需要我们来清空
    p.grad = None
  loss.backward()
  
  #更新grad——————训练到后面的时候减小学习率
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  #状态追踪
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  #记录每次iteration的对应loss
  lossi.append(loss.log10().item())
  with torch.no_grad():
#（6）这部分是为了记录：
#   1.p in parameters里面的p，指的是参数张量们。比如Parameters[0]就是矩阵C，而不是矩阵C内的首个元素
#   2.所以这里实际上计算的是，每一次iteration中，某一个参数矩阵的“变化率”
#   3.并且按照我们当前的设计，整个神经网络里面有19个参数tensor。
#     所以每一次iteration，都会给ud列表加上一个列表I作为元素。
#     列表I内储存的元素是19个参数tensor中每一个参数tensor的(lr*p.grad).std() / p.data.std()).log10()值
#   4.所以ud的维度是i*19。i表示iteration的次数，也就是训练次数
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

  if i >= 1000:
    break # AFTER_DEBUG: would take out obviously to run full optimization




#====================================开始为了调试而打印图像====================================
#（1）这一张看前向信号（激活值）是否健康，具体说是看tanh层的输出有没有过饱和。
#   1.这个直方图的横坐标是：tanh层的输出值，也就是激活值。
#       因为tanh的值域是(-1, 1)，所以横轴大致落在 -1 到 1 之间，形状是"两头被压扁的钟形"。
#   2.纵坐标是：概率密度分布。torch.histogram(..., density=True) 会把直方图归一化成面积=1，
#       这样不同层之间比较的是"形状"，而不是"样本数量"。
#   3.图中有很多条曲线，分别表示每一层tanh输出的激活值分布。
#       注意标签用的是layer在layers列表里的下标（每三件套 Linear,BN,Tanh 一组），所以是 2、5、8、11、14。
#   4.虽然每一tanh层输出是一个tensor（32个样本 × 100个神经元 = 3200个数），
#       但是统计对象是这个tensor里面各个分量的值，也就是把这3200个数展平后一起统计。
#   5.判断过饱和看print里的 saturated，它统计的是 |激活值| > 0.97 的分量所占的百分比。
#       为什么阈值取0.97？因为 tanh'(x) = 1 - tanh(x)**2，
#       当输出值是0.97时，导数只剩下 1 - 0.97**2 ≈ 0.06，
#       也就是梯度传到这里只能按约6%的比例继续往回走；输出到0.99就只剩2%了。
#       所以这个百分比越高，说明越多的神经元被推到了tanh的平缓区，梯度会被压没。
#   6.健康的标志：
#       mean ≈ 0        —— 上一层的BN把激活值正负对称地归一化了
#       std ≈ 0.65      —— 标准正态输入经过tanh后，标准差大约就是0.63；关键是它要逐层基本不变
#       saturated 很低  —— 只有几个百分点，说明绝大多数神经元工作在tanh的线性区
#       反过来，如果std逐层递减，就是"方差塌缩"，是梯度消失的前兆。
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('activation distribution')



#（2）这一张看反向信号（梯度）是否健康。
#   1.数学上，tanh层的叠加会导致梯度的衰减。
#       tanh层的链式法则，会导致输入xin和输出xout有这样的关系：L表示Loss，xin和xout表示向量的一组对应分量
#       dL/dxin = （dL / d xout）* （1 - xout**2），
#       天然乘一个恒小于等于1的数，所以xin的grad一定小于等于xout
#   2.这张图的横坐标是：梯度值。
#   3.纵坐标是：概率密度分布，或者说是频率值
#   4.图中有很多个直线，分别表示每一层tanh输出的梯度分布
#   5.虽然输出是一个tensor，但是统计对象是这些tensor里面各个分量的梯度。
plt.figure(figsize=(20, 4)) # 定义图的大小
legends = []
for i, layer in enumerate(layers[:-1]): #最后一层不参与循环
  if isinstance(layer, Tanh):#判断 layer 是不是 Tanh 类的实例。
    t = layer.out.grad#如果是tanh的layer，那么就取出变量
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('gradient distribution')



#（3）这一张看参数（权重矩阵）的梯度是否健康，也就是"反向传播传到最后，落到参数身上的梯度有多大"。
#   1.横坐标是：参数的梯度值，也就是 p.grad 里面各个分量的值。
#   2.纵坐标是：概率密度分布（同样是density=True归一化过）。
#   3.只挑 ndim == 2 的参数画图，也就是矩阵形状的权重：C 和 6 个 Linear 的 weight，一共7条曲线。
#       BN的gamma/beta是一维的，不画：一是它们的每个分量对应"一个特征"，看分布意义不大；
#       二是它们初值全相同（gamma全1、beta全0），std=0，算前面的比值时会得到inf。
#   4.最关键的是print出来的 grad:data ratio = grad.std() / data.std()，
#       这是一个无量纲的比值，表示"梯度的大小相对于参数本身的大小"。
#       它决定了每一步更新的相对步长，经验上希望它落在 1e-3 附近。
#   5.注意这个梯度是"最后一次"前向-反向算出来的（循环里每步都会重新计算并覆盖），
#       所以它是训练结束那一刻的快照。
#   6.健康的标志：
#       各层梯度的量级不要相差太多（不要出现前面1e-2、后面1e-6这种数量级的落差）；
#       grad:data ratio 在 1e-3 上下，同一数量级即可。
#       如果某个ratio特别大，说明该层学习率过高、会被一步打乱；特别小则说明该层学不动。
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');



#（4）这一张看学习率是否合适——把每一次iteration算出来的"参数变化率"画成随时间的曲线。
#   1.横坐标是：iteration（训练步数，这里因为break，实际只有1001步）。
#   2.纵坐标是：每一个参数矩阵的这个值：log10( std(lr * p.grad) / std(p.data) )。
#       我们按照当前神经网络的设计，假设只有19个参数矩阵。然后其中只有一部分是二维的。假设有7个
#       那么我们这七个tensor，每一次iteration都会计算一次(lr*p.grad).std() / p.data.std()).log10()
#   3.所以整体图显示的就是，有七条线，每一条线都表示一个二维tensor的学习率，关于i的变化趋势
#   3.注意 ud[j][i] 是转置：ud本身是 ud[step][param]，
#       而画图想让"某一个参数随时间的演变"成为一条曲线，所以把参数这一维提到外层。
#   4.只画 ndim == 2 的参数，所以图上只有7条曲线（C 和 6 个 Linear 的 weight）。
#   5.那条黑色的直线是 y = -3 的基准线：log10 ≈ -3 表示每步把权重改动约千分之一（1e-3）。
#       这是 Karpathy 的经验法则——比值稳定贴着 -3，说明学习率合适。
#       远高于 -3（比如 -1）说明学习率太大，参数会被一步打乱、训练震荡；
#       远低于 -3 说明学习率太小，参数几乎不动、学得很慢。
#   6.为什么用log10：不同参数、不同阶段的比值会跨越好几个数量级，取对数后才好在同一张图上比较。
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:#我们要研究的参数张量都是两个维度的。也就是C矩阵和线性层的W，B矩阵
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);
#（5）我们发现，最后一个linear的W矩阵，学习率的值比较高，why？



@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

# put layers into eval mode
for layer in layers:
  layer.training = False
split_loss('train')
split_loss('val')


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # forward pass the neural net
      emb = C[torch.tensor([context])] # (1,block_size,n_embd)
      x = emb.view(emb.shape[0], -1) # concatenate the vectors
      for layer in layers:
        x = layer(x)
      logits = x
      probs = F.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out)) # decode and print the generated word
    
    
    


def normshow(x0):
  
  g = torch.Generator().manual_seed(2147483647+1)
  x = torch.randn(5, generator=g) * 5
  x[0] = x0 # override the 0th example with the slider
  mu = x.mean()
  sig = x.std()
  y = (x - mu)/sig

  plt.figure(figsize=(10, 5))
  # plot 0
  plt.plot([-6,6], [0,0], 'k')
  # plot the mean and std
  xx = np.linspace(-6, 6, 100)
  plt.plot(xx, stats.norm.pdf(xx, mu, sig), 'b')
  xx = np.linspace(-6, 6, 100)
  plt.plot(xx, stats.norm.pdf(xx, 0, 1), 'r')
  # plot little lines connecting input and output
  for i in range(len(x)):
    plt.plot([x[i],y[i]], [1, 0], 'k', alpha=0.2)
  # plot the input and output values
  plt.scatter(x.data, torch.ones_like(x).data, c='b', s=100)
  plt.scatter(y.data, torch.zeros_like(y).data, c='r', s=100)
  plt.xlim(-6, 6)
  # title
  plt.title('input mu %.2f std %.2f' % (mu, sig))

interact(normshow, x0=(-30,30,0.5));



# Linear: activation statistics of forward and backward pass

g = torch.Generator().manual_seed(2147483647)

a = torch.randn((1000,1), requires_grad=True, generator=g)          # a.grad = b.T @ c.grad
b = torch.randn((1000,1000), requires_grad=True, generator=g)       # b.grad = c.grad @ a.T
c = b @ a
loss = torch.randn(1000, generator=g) @ c
a.retain_grad()
b.retain_grad()
c.retain_grad()
loss.backward()
print('a std:', a.std().item())
print('b std:', b.std().item())
print('c std:', c.std().item())
print('-----')
print('c grad std:', c.grad.std().item())
print('a grad std:', a.grad.std().item())
print('b grad std:', b.grad.std().item())



# Linear + BatchNorm: activation statistics of forward and backward pass

g = torch.Generator().manual_seed(2147483647)

n = 1000
# linear layer ---
inp = torch.randn(n, requires_grad=True, generator=g)
w = torch.randn((n, n), requires_grad=True, generator=g) # / n**0.5
x = w @ inp
# bn layer ---
xmean = x.mean()
xvar = x.var()
out = (x - xmean) / torch.sqrt(xvar + 1e-5)
# ----
loss = out @ torch.randn(n, generator=g)
inp.retain_grad()
x.retain_grad()
w.retain_grad()
out.retain_grad()
loss.backward()

print('inp std: ', inp.std().item())
print('w std: ', w.std().item())
print('x std: ', x.std().item())
print('out std: ', out.std().item())
print('------')
print('out grad std: ', out.grad.std().item())
print('x grad std: ', x.grad.std().item())
print('w grad std: ', w.grad.std().item())
print('inp grad std: ', inp.grad.std().item())