#这里很多矩阵运算和python语法还不熟悉。。。。。。
#以后还是得亲自手算一下小样例才行


#========================本篇带领我们手动进行backward，对梯度更新理解更深刻=================
#（1）第一个总结：
#   1.如果前向传播中，从A得到B的过程，需要广播机制，也就是A需要复制自身，来对齐B：
#     那么后续backward从B求A的时候，因为同一个A的元素经过多次复制，运算得到B，也就是一个A的元素是多个B元素的父元素
#     那么backward的时候就需要沿着B的某一个维度进行sum，来累加正确得到被多次使用的A元素的grad

#   2.如果前向传播中，从A得到B的过程，需要求和机制，也就是一个B元素，是多个A元素的子元素
#     比如B_1 = A_1 + A_2 + A_3 +...+ A_n
#     那么backward的时候，就需要广播机制。复制多个B1，然后分别将对B1的grad，直接传递到A_n上。
#     因为B_1对这些A的grad都是1，所以直接将dL / dB1传就行了
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
#===================================初始化数据集和context==================================
# read in all the words
words = open('Neural Network_Zero to hero/makemore/names.txt', 'r').read().splitlines()

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

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

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2]) # 10%
Xte, Yte = build_dataset(words[n2:])     # 10%


#==============================declare utility function=======================
#用来评估我们手算梯度准确性的参数
def cmp(s, dt, t):
    #传入三个参数。s是一个字符串，用来表示当前参数张量是哪一个layer的
    #dt是我们手算的梯度
    #t是Pytorch计算的梯度
  ex = torch.all(dt == t.grad).item()
  #torch.all表示逐元素对比，我们手算的和Pytorch官方计算的grad是否相同。返回值是一个布尔张量
  #.item()表示把布尔tensor的值，也就是布尔值取出。
  #所以ex就是一个Boolean类型的变量
  app = torch.allclose(dt, t.grad)
  #这里判断的就是，是否两个tensor近似相等
  maxdiff = (dt - t.grad).abs().max().item()
  #这里是计算，两个tensor之间，各个元素的最大误差的绝对值。
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
  #整行打印出：参数名、是否精确相等、是否近似相等、最大误差。 


#==============================初始化参数======================================
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
batch_norm_gain = torch.randn((1, n_hidden))*0.1 + 1.0
batch_norm_bias = torch.randn((1, n_hidden))*0.1

# Note: I am initializing many of these parameters in non-standard ways
# because sometimes initializing with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, batch_norm_gain, batch_norm_bias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y



#===============================训练层=============================================
#forward pass层：步骤拆分的非常细腻，方便我们一步步手动backward
#embedding层
emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors

# Linear layer 1
hpre_batch_norm = embcat @ W1 + b1 # hidden layer pre-activation：送进BN之前的预激活

# BatchNorm layer
#原本算式：x是BtachNorm层输入的tensor
#   BatchNorm_output = batch_norm_gain * (x - xmean) / torch.sqrt(xvar + self.eps) + batch_norm_bias
batch_norm_meani = 1/n*hpre_batch_norm.sum(0, keepdim=True)#输入tensor内各元素的平均值
batch_norm_diff = hpre_batch_norm - batch_norm_meani
batch_norm_diff2 = batch_norm_diff**2#为了计算输入tensor的方差，而存在的中间量
batch_norm_var = 1/(n-1)*(batch_norm_diff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
batch_norm_inverse_std = (batch_norm_var + 1e-5)**-0.5 # inv=inverse(倒数)：这里是 1/sqrt(var+eps)，也就是标准差的倒数
batch_norm_raw = batch_norm_diff * batch_norm_inverse_std
hpre_batch_norm_out = batch_norm_gain * batch_norm_raw + batch_norm_bias # BN的输出，也就是tanh的输入

# Non-linearity
h = torch.tanh(hpre_batch_norm_out) # hidden layer

# Linear layer 2
logits = h @ W2 + b2 # output layer

#拆解交叉熵的计算(The outcome is just the same as the code： F.cross_entropy(logits, Yb))
#原式子：式子：loss = - (1/N) * Σ_i log( exp(logits[i, Yb[i]]) / Σ_j exp(logits[i, j]) )
#（0）巩固一下：logits[i, Yb[i]]是高级查表：
#   1.i是一个整数，yb[i]也是一个整数。
#     所以整体就是取出样本i，输入神经网络后，输出正确的ch，也就是yb[i]这个数字对应的ch的概率

#（1）为什么这里要减去最大值？
#   先看这里是怎么减的：
#       1.logits的维度是(batch_size, vocab_size)，也就是(32,27)
#       2.logits.max(1, keepdim=True).values   # dim=1 → 每行一个最大值，形状 (32,1)
#       3.所以是Batch里面的每一个样本，减去样本内27个分量的最大值

#   减去最大值的好处：
#       1.防止溢出————logits的值可能很大，取了exp(logits[i, Yb[i]])可能溢出
#       2.比如[100,100.....,100]减去最大值后，最大的那一分量变成0，其余都小于0，就不可能溢出了

#   并且减去同一个样本内，各个分量都减去最大值，不影响最终该样本的Loss：
#       1.一个样本i对应的Loss计算式：log( exp(logits[i, Yb[i]]) / Σ_j exp(logits[i, j])
#       2.那么j，就表示这个样本i，对应神经网络输出的长度27的tensor内，第j个分量
#       3.假设减去最大值k，就会变成：
#           log( exp( logits[i, Yb[i]] - k ) / Σ_j exp( logits[i, j] - k )
#       4.先看分母：Σ_j exp( logits[i, j] - k )：
#           若logits[i,j]记为s_j，
#           那么分母就是：e的s_1-k次方 + e的s_1-k次方 +...+ e的s_27-k次方
#           就等于（e的s_1次方 + e的s_2次方 + e的s_27次方）/ e的k次方
#       5.再看分子：exp( logits[i, Yb[i]] - k )
#           等价于：e的t-k次方，也就是e的t次方 / e的k次方
#       6.上下消去了最大值k，所以总的来说，该操作对计算没有任何影响           
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)#对每一个样本内的元素进行求和
counts_sum_inverse = counts_sum**-1 # inv=inverse：counts_sum的倒数。
probs = counts * counts_sum_inverse
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
#（2）这里的语法：为什么可以放入一个range(n)来查表？
#   1.logprobs 形状是 (n, vocab_size)，也就是 (32, 27)。n就是batch_size
#   2.给一个二维张量两个索引的时候，tensor会智能的把他们配对：效果等价如下。是一种高级索引
#   [
#        logprobs[0, Yb[0]],
#        logprobs[1, Yb[1]],
#        ...
#        logprobs[n-1, Yb[n-1]]
#    ]

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inverse, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpre_batch_norm_out, batch_norm_raw,
          batch_norm_inverse_std, batch_norm_var, batch_norm_diff2, batch_norm_diff, hpre_batch_norm, batch_norm_meani,
          embcat, emb]:
  t.retain_grad()
  #表示把所有的中间参与过运算的tensor都保留其grad。方便我们后续对比
  #本来应该只有w参数们，作为叶子tensor，才会被保留grad               
loss.backward()



#======================练习一：手动实现该神经网络所有的backward================
#前置：矩阵对矩阵求导的实质：
#   1.就是澄清一个易混淆的点：Loss是一个矩阵，Logits也是一个矩阵，那么Loss对Logits求偏导是什么意思？
#   2.就是对于Loss里面每一个分量a，对Logits里面每一个和a有关的变量b，求偏导。
#   3.所以矩阵对矩阵求导，其实是对矩阵内具体的值求导

#核心：对于一个tensor内的value对象来说：
# 对应一个value，也就是tensor内的一个元素而言，
# backward等于，接收并且累加它的子value传给它的grad，
# 且计算经由自己传导给自己父value们的grad并且传给父元素
# 子元素，就是forward pass的过程中，会使用本value的输出作为其输入计算其data的value
# 负元素，就是在forward pass中，作为输入计算本value的data的value

#一、第一步：从Loss，来backward到logits
#（1）先从Loss，backward到logprob：
#   1.probs矩阵，是这样的：（Batch_size,vocab_size）。
#     表示每一个样本，输入神经网络之后，神经网络根据这个输入，猜测下一个输出的ch的概率分布
#     也就是下一个输出是这27种ch中的每一种的概率是多大
#     比如probs[1,1]就是，第一个样本，输入神经网络之后，神经网络预测输出下一个ch字符是a的概率
#   
#   2.但是，对于logprob矩阵，每一个样本，都只有一个元素被拿去计算Loss了。一共被拿了Batch_size个
#     详见：loss = -logprobs[range(n), Yb].mean()
#     所以说，logprob矩阵内，只有一部分元素，和Loss有关。只有这些有关的元素，才会在backward过程被遍历到
#     别的元素的grad都默认是0，不会受到任何影响。因为它们的改变并不会导致Loss的改变，自然grad应该是0
#   
#   3.假设，a，b，c.....z都是和Loss有关的，那么Loss实际上就是：Loss = -（1.0/Batch_size）*（a+b+c+...+z）
#     也就是：Loss = （-1/32）a + （-1/32）* b + ... + （-1/32）* z
#   
#   4.所以我们就找出，logprobs矩阵内，被选择过的
derivative_logprobs = torch.zeros_like(logprobs)
derivative_logprobs[range(n), Yb] = -1.0/n

#（2）从logprobs，推回到probs
#     1.只有derivative_logprobs[range(n), Yb]，这部分选择的元素，才是有grad，才需要向上追溯
#     2.那么对于每一个derivative_logprobs[range(n), Yb]选中的元素，都对应probs[range(n), Yb]选中的元素唯一对应
#     3.所以假设a是logprobs的一个有关元素，在probs中唯一对应的有关元素是b
#       所以： a = lnb 。
#       那么：da / db = 1.0/b
derivative_probs = (1/probs) * derivative_logprobs


#（3）从probs，推回到counts和counts_sum_inverse。
#     1.其实counts_sum_inverse也是从counts来的
#     2.counts是一个(batch_size,vocab_size）的tensor。每一个元素的含义是，一个样本内的一个分量的相对频率
#     3.我们知道porbs[1,1], 就是第一个样本，输入神经网络之后，神经网络预测输出下一个ch字符是a的概率
#     4.就是由counts[1,1]乘counts_sum_inverse[0]得来的
#     5.counts_sum_inverse[0]，就是counts[0]这个一维tensor内，各个分量求和，然后取倒数。
#     6.counts_sum_inverse这个矩阵规格是[32,1].
#     7.但是counts和counts_sum_inverse规格不同，后者只有一列。所以会触发复制。复制出27行变成[32,27]的矩阵
#       然后再进行element-wise，逐元素乘法
#     7.所以逐元素表达式：若probs[i][j] = a ; counts[i][j] = b , counts_sum_inverse[i] = c
#       我们有：a = bc
#       所以 da / dc = b
derivative_counts_sum_inverse = (counts*derivative_probs).sum(1,keepdim=True)
#注意：counts*probs是对应元素相乘，不是矩阵乘法
#（4）为什么这里还需要补一个.sum(1,keepdim=True)？
#     1.回忆前向传播的时候：
#       如果当前层的一个Value，被多个下一层的Value使用为输出。那么当backward的时候，传回的grad会叠加。
#       比如：
#         当前层有a，下一层分别是：b = s + a， c = at
#         那么dL / da = （dL/db）*（db/da）+（dL/dc）*（dc/da） 
#     2.所以对于counts*derivative_probs，这个操作得到的矩阵 K 仍然是(batch_size,vocab_size)大小的
#       probs的第[i]行的所有元素，都有counts_sum_inverse[0]这个元素作为父元素。
#       显然矩阵K的[i][j]元素，就是probs[i][j]关于父元素counts_sum_inverse[i]的grad。
#       所以我们要对k，沿着行，跨过列来求和，也就是把grad都累加到这一行的共有父元素counts_sum_inverse[i]
#       
derivative_counts = derivative_probs* counts_sum_inverse
#（5）这里求counts的derivative。
#   对于a=probs[i][j]，一个父元素是b=counts[i][j],另一个就是c=counts_sum_inverse[i]。
#   所以dL / db = （dL / da） * （da / db）
#   并且（da / db）= c
#   很显然得到上面式子
#
#（6）既然counts_sum_inverse也是从counts来的，那为什么不直接得到counts？还要计算counts_sum_inverse？
#   因为counts被使用了两次，它有两个直接子value矩阵，一个是probs，一个是counts_sum = counts.sum(1, keepdim=True)
#   所以要等它所有的子value矩阵，把全部grad都返回给它身上，才可以真正得到dL / d counts



#（7）从counts_sum_inverse得到counts：
#   1.先从counts_sum_inverse 得到 counts_sum：
#     若y = 1/x，那么dy / dx = -1/（x**2）
#   2.并且counts_sum_inverse[i][j]的父元素有且只有counts_sum[i][j]，直接逐元素计算
derivative_counts_sum = -derivative_counts_sum_inverse * counts_sum_inverse**2
#   3.再从counts_sum得到counts：父子关系如下：
#     counts_sum[i] = counts[i][1] + counts[i][2] +... counts[i][27]
#   4.很显然，它要传给每个父元素的grad都是1
derivative_counts += torch.ones_like(counts)*derivative_counts_sum


#（8）从counts到Norm_logits:
#   1.counts和norm_logits大小都是(batch_size, vocab_size)，在这个例子就是（32,27）
#   2.对于a = counts[i][j], b = logits[i][j], 我们有 a = e的b次方
#   3.所以da / db = e的b次方，也就是a。所以 dL / db = （dL / da）*（da / db）
derivative_norm_logits = counts*derivative_counts


#（9）从norm_logits到Logits：
#   1.norm_logits，Logits都是(32,27)，即(batch_size, vocab_size)。Logits_maxes是（32,1）
#   2.对于a = norm_logits[i][j]，b = logits[i][j]， c = logits_maxes[i]，我们有a = b - c
#   3.所以 a 有两个父元素，一个是b，一个是c
#   4.并且对于同一行的a，都有一个共同父元素c，所以a们对c的grad要累加
#   4.先计算logits_maxes的偏导：dL / dc = (dL / da) * (da / dc) 
derivative_logits_maxes = (-1)*derivative_norm_logits.sum(1,keepdim=True)
#   5.再计算Logits的偏导：dL / db = (dL / da) * (da / db) 
#   clone()：不写的话 derivative_logits 和 derivative_norm_logits 是同一个tensor，
#   求导发现d norm_logits / d Logits = 1，所以L对这两个tensor各个元素的导，是一一对应相等的。
#   但是直接用 = 号会有bug。因为这个本质是赋值操作，也就是添加指针指向同一个对象，所以需要clone一份这个对象
derivative_logits = derivative_norm_logits.clone()

#（10）我们需要从Logits_max计算Logits的grad：
#   1.我们有a = Logits_max[i], 和b_j = Logits[i][j], 并且j in range of (0,batch_size-1)
#   2.a = max（b_j） ——————我不知道这个函数如何求导
#   3.呃呃呃，Andrej说其实很简单，假设b就是Logits[i]这一行的最大值。那么其实就是 a = b，传递的grad就是1
#   4.并且值得注意的是，a的父元素只有b这一个，grad仅仅只会传递给这一行的最大值
#     所以采用一位热位编码，只在每一行的被选取作为最大值的位置，放1，再乘上层传下来的grad矩阵，就能限制grad只传给最大值了
derivative_logits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * derivative_logits_maxes


#============================第二层线性层============================
#(1)线性层实质：Logits = H @ W2 + B2
#   1.各矩阵规格：
#     Logits为（Batch_size，vocab_size）
#     H 为 （Batch_size, n_hidden）
#     W2为 （n_hidden,vocab_size）
#     B2为 （vocab_size，1），但是由于广播机制，相当于里面的每一个元素都被重复利用了
#   
#    假设我们有C = A@B + E计算，我们希望求C对A的导
#    [ 矩阵计算展开 ]

#    [ a11  a12 ]   [ b11  b12 ]   [ e1  e2 ]   [ c11  c12 ]
#    [ a21  a22 ] @ [ b21  b22 ] + [ e1  e2 ] = [ c21  c22 ]

#    展开得到：
#    c11 = a11*b11 + a12*b21 + e1
#    c12 = a11*b12 + a12*b22 + e2
#    c21 = a21*b11 + a22*b21 + e1
#    c22 = a21*b12 + a22*b22 + e2

#    [ 梯度反向传播求导 ]
#    dL/da11 = (dL/dc11)*b11 + (dL/dc12)*b12
#    dL/da12 = (dL/dc11)*b21 + (dL/dc12)*b22
#    dL/da21 = (dL/dc21)*b11 + (dL/dc22)*b12
#    dL/da22 = (dL/dc21)*b21 + (dL/dc22)*b22

#    [ 写成矩阵形式（最终结论） ]
#    [ dL/da11   dL/da12 ]   [ dL/dc11   dL/dc12 ]   [ b11   b21 ]
#    [ dL/da21   dL/da22 ] = [ dL/dc21   dL/dc22 ] @ [ b12   b22 ]
#    即：dL/dA = dL/dC @ B^T
#    这里的n_hidden参数是64
#（2）Logits = H @ W2 + B2，W2的规格是(n_hidden, vocab_size)
#   所以 dL/dH = dL/dLogits @ W2.T，必须转置，否则矩阵乘法维度对不上
derivative_h = derivative_logits @ W2.T
derivative_W2 = h.T @ derivative_logits 
#   注意：b2的规格是(vocab_size,)，在前向里被广播到整个batch，所以要求的是
#   “上层梯度” derivative_logits 的跨样本和，不是前向的 logits.sum(0)（那是数据，不是梯度）
derivative_B2 = derivative_logits.sum(0)
#这里我忘了B2的规格，应该是（1，n_hidden），表示对于一个样本的所有激励值加上bias。
#所以backward的时候，需要跨样本，沿着同一个特征向量分量求和


#==============================激活函数层===============================
derivative_hpreact = (1-h**2)*derivative_h

#=============================batch_norm层================================
#（1）对这个算式backward：hpre_batch_norm_out = batch_norm_gain * batch_norm_raw + batch_norm_bias
#   注意：传进本层的上层梯度是 derivative_hpreact（tanh那一层已经算好了），不是 derivative_h
derivative_batch_norm_gain = (derivative_hpreact * batch_norm_raw).sum(0,keepdim=True)
#这里我忘了，gain和bias都是（1，n_hidden），表示对于一个样本的所有激励值加上各自bias，并且乘各自的gain参数
derivative_batch_norm_raw = (batch_norm_gain * derivative_hpreact)
derivative_batch_norm_bias = (derivative_hpreact).sum(0,keepdim=True)

#（2）目标：batch_norm_raw = batch_norm_diff * batch_norm_inverse_std
#   1.规格：batch_norm_raw是(batch_size,n_hidden)，表示每一个样本的每一个元素，都经过了初步normalize
#           batch_norm_diff同上。表示每一个样本的每一个向量，都减去了跨样本的该向量平均值，得到的diff矩阵
#           batch_norm_inverse_std是（1，n_hidden），表示每一个样本的每一个向量，都跨样本求出了std，然后倒置
derivative_batch_norm_diff = derivative_batch_norm_raw * batch_norm_inverse_std
#   2.这里是“先逐元素相乘、再跨样本求和”：dL/d inv_std_j = Σ_i (dL/d raw_ij) * diff_ij
#     不能写成 derivative_batch_norm_raw * batch_norm_diff.sum(0,keepdim=True)
derivative_batch_norm_inverse_std = (derivative_batch_norm_raw * batch_norm_diff).sum(0,keepdim=True)
#   3.inverse_std = (var + eps)**-0.5，所以 d inv_std / d var = -0.5*(var+eps)**-1.5
#     这里用的是 batch_norm_var（不是 batch_norm_inverse_std），而且这个量其实是 dL/dvar
derivative_batch_norm_var = (-0.5*(batch_norm_var+1e-5)**(-1.5)) * derivative_batch_norm_inverse_std

#（3）目标：batch_norm_var = 1/(n-1)*(batch_norm_diff2).sum(0, keepdim=True)。n就是Batch_size
#         batch_norm_diff2 = batch_norm_diff**2
#
#   1.假设var是1*2，那么显然diff2是2*2。那么 var11 = 1/（n-1）* (diff2_11 + diff2_21)
#     所以dvar / d diff2 = 一个2*2的矩阵，每一个元素都是1/（n-1）
derivative_batch_norm_diff2 = torch.ones_like(batch_norm_diff2)*(1/(n-1)) * derivative_batch_norm_var
#   2.diff2 = diff**2，所以 d diff2 / d diff = 2*diff，用的是 batch_norm_diff（不是 batch_norm_diff2）
derivative_batch_norm_diff += 2*batch_norm_diff*derivative_batch_norm_diff2

#（4）目标：先batch_norm_diff = hpre_batch_norm - batch_norm_meani,这里meani涉及了广播机制
#          再batch_norm_meani = 1/n*hpre_batch_norm.sum(0, keepdim=True)#输入tensor内，跨样本的分量的平均值
#   1.clone()是必须的：若写成 derivative_hpre = derivative_batch_norm_diff，
#     两个名字会指向同一个tensor，下面的 += 会就地改掉 derivative_batch_norm_diff
derivative_hpre = derivative_batch_norm_diff.clone()
#   2.diff = hpre - meani，diff对meani的导数是-1，所以要跨样本累加
derivative_batch_norm_meani = (-derivative_batch_norm_diff).sum(0,keepdim=True)
#   3.meani = 1/n * hpre.sum(0)，每个样本的hpre都要分到 (1/n)*dL/dmeani
derivative_hpre += torch.ones_like(hpre_batch_norm)*(1.0/n)*derivative_batch_norm_meani


#=====================第一线性层=================================
#（1）线性层：hpre_batch_norm = embcat @ W1 + b1
#   1.各矩阵规格：
#     hpre_batch_norm 为 (Batch_size, n_hidden) = (32, 64)
#     embcat          为 (Batch_size, n_embd*block_size) = (32, 30)
#     W1              为 (n_embd*block_size, n_hidden) = (30, 64)
#     b1              为 (n_hidden,) = (64,)
#   2.矩阵求导的结论和上面第二线性层一模一样，只是字母换了：
#     若 C = A @ B，则 dL/dA = dL/dC @ B.T，dL/dB = A.T @ dL/dC
derivative_embcat = derivative_hpre @ W1.T
derivative_W1 = embcat.T @ derivative_hpre
#   3.b1也是被广播的（一个(64,)的向量被加到32个样本上），所以要把上层梯度沿着样本维度sum掉
derivative_b1 = derivative_hpre.sum(0)
#（2）顺带说一个有趣的事实：b1其实是个“废物参数”
#   因为hpre_batch_norm马上就要送进BatchNorm，而BN第一步就是减去跨样本均值batch_norm_meani。
#   给每个分量加上b1，只是让整列整体平移，这个平移量在“减去均值”的瞬间就被彻底消掉了，
#   所以b1对最终Loss没有任何影响（它的grad虽然能算出来，但学不到任何东西）。


#=============================view层（拼接）===================================
#（1）embcat = emb.view(Batch_size, -1)，这是一个纯粹的“形状变换”：
#   1.它不改变内存里元素的存放顺序，只是把(32, 3, 10)重新解释成(32, 30)
#   2.所以backward就是它的逆操作：把(32, 30)再view回(32, 3, 10)
#   3.两种形状同一位置的元素一一对应，所以既不需要sum，也不需要复制
derivative_emb = derivative_embcat.view(emb.shape)


#=============================embedding层=====================================
#（1）emb = C[Xb]，这一步是“查表”：
#   1.C的规格是(vocab_size, n_embd)=(27, 10)，Xb的规格是(Batch_size, block_size)=(32, 3)
#   2.输出的emb[i][j]就等于C的第Xb[i][j]行，也就是把C里的某一行“复制”到emb里
#   3.所以C的每一行都会被复制很多次（同一个字符在同一个batch里可能出现多次）
#   4.于是backward的时候，C的每一行要把所有“复制过它的位置”传回来的grad累加起来
#（2）所以这里就直接用一个双重循环，把“累加”这件事老老实实写出来：
#   注意要用zeros_like先开一块新内存，不能直接在C.grad上累加，
#   因为我们要自己算一份梯度，去和PyTorch算的C.grad做对比
derivative_C = torch.zeros_like(C)
for i in range(n):
    for j in range(block_size):
        character_index = Xb[i][j]                  # 这个位置上查表用的字符编号
        derivative_C[character_index] += derivative_emb[i][j]
        #这里emb是一个三维tensor，所以每一次加回去给C，都是加一个维度的tensor。
        #也就是每一次加给C，都是加上一个字符对应的10维向量
        


#======================练习二：用cmp验证我们手算的梯度========================
#（1）使用说明：cmp(s, dt, t)会把我们手算的dt和 t.grad（PyTorch算的）作对比
#   1.所以传进来的t必须是前面retain_grad()过的中间张量，或者是requires_grad的参数
#   2.不要强求 exact=True：浮点数的加减乘除结合顺序不同，结果就会有1e-9左右的差异，
#     只要 approximate=True、maxdiff 在1e-8量级，就说明我们的推导是正确的
#（2）从最后一层往回，逐个对比（包括参数和所有中间张量）：
cmp('C',                      derivative_C,                      C)
cmp('emb',                    derivative_emb,                    emb)
cmp('embcat',                 derivative_embcat,                 embcat)
cmp('W1',                     derivative_W1,                     W1)
cmp('b1',                     derivative_b1,                     b1)
cmp('hpre_batch_norm',        derivative_hpre,                   hpre_batch_norm)
cmp('batch_norm_meani',       derivative_batch_norm_meani,       batch_norm_meani)
cmp('batch_norm_diff',        derivative_batch_norm_diff,        batch_norm_diff)
cmp('batch_norm_diff2',       derivative_batch_norm_diff2,       batch_norm_diff2)
cmp('batch_norm_var',         derivative_batch_norm_var,         batch_norm_var)
cmp('batch_norm_inverse_std', derivative_batch_norm_inverse_std, batch_norm_inverse_std)
cmp('batch_norm_raw',         derivative_batch_norm_raw,         batch_norm_raw)
cmp('batch_norm_gain',        derivative_batch_norm_gain,        batch_norm_gain)
cmp('batch_norm_bias',        derivative_batch_norm_bias,        batch_norm_bias)
cmp('hpre_batch_norm_out',    derivative_hpreact,                hpre_batch_norm_out)
cmp('h',                      derivative_h,                      h)
cmp('W2',                     derivative_W2,                     W2)
cmp('b2',                     derivative_B2,                     b2)
cmp('logits',                 derivative_logits,                 logits)
cmp('logit_maxes',            derivative_logits_maxes,           logit_maxes)
cmp('norm_logits',            derivative_norm_logits,            norm_logits)
cmp('counts',                 derivative_counts,                 counts)
cmp('counts_sum',             derivative_counts_sum,             counts_sum)
cmp('counts_sum_inverse',     derivative_counts_sum_inverse,     counts_sum_inverse)
cmp('probs',                  derivative_probs,                  probs)
cmp('logprobs',               derivative_logprobs,               logprobs)





