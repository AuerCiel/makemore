#=================== 训练 lec3 的 MLP 并采样生成名字（支持GPU加速） ===================
#（1）为什么不直接 import lec3_AdjustParameters_part2？
#   1.那个文件在模块顶层就直接跑训练循环、画图，还调用了ipywidgets的interact，
#     一旦import就会把整套东西重新执行一遍（而且会卡在widget上）
#   2.它的目录名"Neural Network_Zero to hero"带空格和横杠，本身也不是合法的包名
#   3.所以这里把数据集准备和几个layer类重新写了一遍，结构、超参、随机种子都和lec3保持一致
#（2）相对lec3，本文件做了四处改动：
#   1.所有参数和数据都搬到 device 上（cuda / mps / cpu 自动选择）
#   2.去掉了训练时的调试开销：layer.out.retain_grad() 和 ud 统计，它们会明显拖慢训练
#   3.采样函数 use() 自己负责切换BatchNorm的eval/training模式，并在结束时还原
#   4.默认的batch_size从32改成256、步数从20万改成2.5万，两者相乘都是640万个样本，
#     看过的数据量不变，但每次更新用256个样本、梯度更稳，步数只要1/8，跑起来快得多
#     （想完全复现lec3的动态：--batch_size 32 --max_steps 200000）
#（3）本文件不会修改任何lec3的文件
#（4）实测性能（RTX 4070 Laptop，batch_size=32，n_hidden=1000，共420万个参数）：
#   1.每一步约4.4ms：前向1.35ms + 反向2.56ms + 参数更新0.50ms
#   2.但每一步的计算量只有约0.8GFLOP，按这张卡的算力只要0.2ms左右就能算完，
#     说明瓶颈不在计算，而在"算子数量太多、每个kernel的启动与调度开销"
#   3.这也解释了为什么换到GPU只比CPU快两倍左右：batch只有32、宽度只有1000，单次计算实在太小
#   4.想再快，要么减少kernel数量（torch.compile + inductor，
#     但Windows上需要额外安装triton-windows，实测默认后端会报TritonMissing），
#     要么加大batch_size：本文件默认已经这么做了（见上面第4条改动），
#     batch从32加到256以后，单步里那些固定开销被摊薄，整体快了不止两倍
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F


#=================================== 命令行参数 ===================================
def parse_arguments():
  parser = argparse.ArgumentParser(description='训练lec3的MLP并采样生成名字')
  #（1）默认配置：batch_size=256、max_steps=25000
  #   两者相乘是640万个样本，和lec3（batch 32 × 20万步）看过的样本总量一致，
  #   好处是每次更新用256个样本、梯度更稳，步数只需要上一版的1/8，跑得快很多
  #   代价是更新次数变少，训练动态不再和lec3逐位等价
  parser.add_argument('--max_steps', type=int, default=25000,
                      help='训练步数(默认25000,和batch_size=256配套)')
  parser.add_argument('--batch_size', type=int, default=256,
                      help='每个minibatch的样本数(默认256;想复现lec3就设32并配--max_steps 200000)')
  parser.add_argument('--number_of_names', type=int, default=10,
                      help='交互式采样里“直接回车”时生成的名字个数(默认10)')
  parser.add_argument('--device', type=str, default='auto',
                      help='训练设备:auto / cuda / mps / cpu(默认auto)')
  return parser.parse_args()


arguments = parse_arguments()


#=================================== 设备选择 ===================================
def choose_device(requested_device):
  if requested_device != 'auto':
    return torch.device(requested_device)
  if torch.cuda.is_available():                 # NVIDIA显卡
    return torch.device('cuda')
  if torch.backends.mps.is_available():         # 苹果芯片的GPU
    return torch.device('mps')
  return torch.device('cpu')


device = choose_device(arguments.device)
print(f'使用设备: {device}')
if device.type == 'cuda':
  print(f'显卡型号: {torch.cuda.get_device_name(0)}')


#=================================== 数据集准备 ===================================
#（1）用__file__来定位names.txt：这样不管在哪个目录下运行脚本，都能找到数据文件
data_path = Path(__file__).resolve().parent / 'names.txt'
words = data_path.read_text(encoding='utf-8').splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

block_size = 5 # 上下文长度：读入多少个上文，来预测下一个char


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


#（2）和lec3用同样的随机种子，保证切分出来的训练/验证/测试集完全一致
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%


#=================================== 线性层 ===================================
class Linear:

  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
#   良好的初始化Weight矩阵，确保数值分布合理
    self.bias = torch.zeros(fan_out) if bias else None
#   一开始不设置任何偏置——————我们希望线性层的输出，大致符合标准正态分布

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


#=================================== BatchNorm层 ===================================
class BatchNorm1d:

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True

    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True)   # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
    self.out = self.gamma * xhat + self.beta

    #在训练的过程中不断更新逼近全局mean
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]


#=================================== 激活函数层 ===================================
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []


#=================================== 初始化参数 ===================================
n_embd = 30     # 字符嵌入向量的维度
n_hidden = 1000 # 隐藏层神经元个数
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((vocab_size, n_embd), generator=g)
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]

#（1）最后一层BN的gamma乘0.1：让网络一开始输出接近均匀分布，而不是"自信地瞎猜"
#   所以初始loss大约是3.29，而不是没做这个压制时的3.80
#   注：lec3里还有一段 layer.weight *= 1.0，那是保留作警示的空操作，这里就不写了
with torch.no_grad():
  layers[-1].gamma *= 0.1

#（2）把嵌入矩阵、所有层的参数、BatchNorm的running统计量都搬到device上
#   必须在设置requires_grad之前搬：已经是叶子且要梯度的张量，不好再做in-place替换
C = C.to(device)
for layer in layers:
  if isinstance(layer, Linear):
    layer.weight = layer.weight.to(device)
    if layer.bias is not None:
      layer.bias = layer.bias.to(device)
  elif isinstance(layer, BatchNorm1d):
    layer.gamma = layer.gamma.to(device)
    layer.beta = layer.beta.to(device)
    #running_mean / running_var不是参数，但eval模式和采样都要用到，也必须搬过去
    layer.running_mean = layer.running_mean.to(device)
    layer.running_var = layer.running_var.to(device)

#（3）把所有参数收集到一个列表里，统一做梯度清零和更新
parameters = [C] + [p for layer in layers for p in layer.parameters()]
print('参数总量:', sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True

#（4）数据集也搬到device上，这样每个step就不需要反复做CPU->GPU的拷贝
Xtr,  Ytr  = Xtr.to(device),  Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
Xte,  Yte  = Xte.to(device),  Yte.to(device)


#=================================== 开始训练 ===================================
batch_size = arguments.batch_size
index_chunk_size = 1024 # 一块索引够用多少步（默认256*1024个int64，约占2MB显存）
lossi = []
start_time = time.time()

for step in range(arguments.max_steps):

  #（1）取一个minibatch
  #   1.如果每一步都做一次 torch.randint(...).to(device)，就等于每步做一次CPU->GPU拷贝，
  #     而普通内存(pageable)的拷贝会同步整条CUDA流，CPU没办法提前把后面的kernel排进队列，
  #     实测这一步会让训练慢好几倍
  #   2.所以改成"整块抽、整块搬、按步切"：每index_chunk_size步才做一次拷贝，
  #     其余步骤只是从GPU上的这块索引里切一片（切片是view，既不拷贝也不同步）
  #   3.为什么不影响复现性：这个generator是确定性的，
  #     "一次抽N*t个"和"分t次每次抽N个"得到的序列完全相同（已实测验证），
  #     所以minibatch的选取顺序和lec3一模一样
  if step % index_chunk_size == 0:
    chunk_steps = min(index_chunk_size, arguments.max_steps - step)
    index_chunk = torch.randint(0, Xtr.shape[0], (chunk_steps * batch_size,), generator=g).to(device)

  chunk_offset = (step % index_chunk_size) * batch_size
  ix = index_chunk[chunk_offset : chunk_offset + batch_size]
  Xb, Yb = Xtr[ix], Ytr[ix]

  #（2）前向传播
  emb = C[Xb]
  x = emb.view(emb.shape[0], -1)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb)

  #（3）反向传播
  for p in parameters:
    p.grad = None
  loss.backward()

  #（4）更新参数：训练后1/4的阶段把学习率降到1/10（和lec3的150000/200000一致）
  #   注：batch变大后理论上的经验做法是按比例调大学习率，但这里故意保持lec3的0.1不动，
  #   因为它对256的batch依然是安全的，少引入一个变量
  learning_rate = 0.1 if step < 0.75 * arguments.max_steps else 0.01
  for p in parameters:
    p.data += -learning_rate * p.grad

  #（5）记录loss，并且定期打印进度和耗时
  #   注意：不要在每一步都调用 .item()！
  #   .item() 会把GPU的异步执行强行同步：CPU必须等GPU把这一步算完才能继续，
  #   于是每一次kernel启动的开销都被完整暴露出来（实测能慢三倍以上）
  #   所以这里每100步才同步一次，用来记录loss曲线
  if step % 100 == 0 or step == arguments.max_steps - 1:
    lossi.append(loss.item())

  if step % 10000 == 0 or step == arguments.max_steps - 1:
    elapsed = time.time() - start_time
    print(f'{step:7d}/{arguments.max_steps:7d}: loss {loss.item():.4f} | 已用时 {elapsed:.1f}s')


#=========================== 评估train/val/test的loss ===========================
evaluation_batch_size = 4096

#（1）把所有BatchNorm层切到指定模式
def set_batch_norm_mode(training):
  for layer in layers:
    if isinstance(layer, BatchNorm1d):
      layer.training = training

#（2）记录当前每个BatchNorm层的模式，方便结束后还原
def save_batch_norm_modes():
  return [(layer, layer.training) for layer in layers if isinstance(layer, BatchNorm1d)]

#（3）把之前记录的模式还原回去
def restore_batch_norm_modes(saved_modes):
  for layer, training in saved_modes:
    layer.training = training


@torch.no_grad()
def split_loss(split):
  x, y = {
    'train': (Xtr, Ytr),
    'val':   (Xdev, Ydev),
    'test':  (Xte, Yte),
  }[split]

  #（4）为什么要分块算？val集有2万多个样本，一次性喂进去的话，
  #   5个隐藏层的激活值各占几百MB显存，会白白吃满显存
  #   分块时用reduction='sum'再按样本数平均，结果和一次性算完全相同
  total_loss = 0.0
  for start in range(0, x.shape[0], evaluation_batch_size):
    Xb = x[start:start + evaluation_batch_size]
    Yb = y[start:start + evaluation_batch_size]
    emb = C[Xb]
    out = emb.view(emb.shape[0], -1)
    for layer in layers:
      out = layer(out)
    total_loss += F.cross_entropy(out, Yb, reduction='sum').item()

  average_loss = total_loss / x.shape[0]
  print(f'{split:5s} loss: {average_loss:.4f}')
  return average_loss


#（5）评估必须在eval模式下做：BN要用训练时累积的running统计量，
#   否则"整个验证集一起算batch统计量"会把答案泄漏进BN里，算出来的loss虚低
saved_modes = save_batch_norm_modes()
set_batch_norm_mode(False)
split_loss('train')
split_loss('val')
split_loss('test')
restore_batch_norm_modes(saved_modes)


#=================================== 采样：生成名字 ===================================
#（0）固定一个采样专用的随机种子，让整个会话的结果都可复现
#   注意：这个生成器只在这里创建一次（而不是每次调用use都重建），
#   于是交互模式下每生成一轮，它就接着上一条随机流往后走，
#   所以连续输入两次10，得到的是两组不同的名字，而不是一模一样的两组
sample_generator = torch.Generator().manual_seed(2147483647 + 10)


def use(number_of_names=10):

  #（1）进入时把BatchNorm切成eval模式，并记下原来的模式，退出时还原
  #   为什么必须要eval？采样时每次前向传播只有1个样本，BN若在training模式下，
  #   就会拿"这一个样本自己"当均值，于是 xhat = (x - x)/sqrt(0+eps) = 0，
  #   BN的输出只剩beta，和输入完全无关，整条链路的信息被抹掉，生成的全是乱码
  saved_modes = save_batch_norm_modes()
  set_batch_norm_mode(False)

  #（2）采样用的随机数生成器来自模块层面的sample_generator：
  #   它只在启动时种一次种子，每调用一次use就往前推进一段，
  #   所以连续两次输入相同个数，得到的是两组不同的名字

  try:
    with torch.no_grad(): # 采样不需要梯度，关掉可以省显存、也更快
      for name_index in range(number_of_names):

        #（3）每个名字都从"名字的开头"开始：上下文是block_size个'.'（'.'的编号是0）
        context = [0] * block_size
        generated_characters = []

        while True:
          #（4）前向传播：把上下文喂进网络，得到27个字符各自的logits
          emb = C[torch.tensor([context], device=device)]
          x = emb.view(emb.shape[0], -1)
          for layer in layers:
            x = layer(x)
          probabilities = F.softmax(x, dim=1)

          #（5）把概率搬到CPU再采样：
          #   torch.multinomial的generator必须和tensor在同一个设备上，
          #   而固定用CPU上的generator，可以保证CPU和GPU训出来的模型生成的名字一致
          character_index = torch.multinomial(probabilities[0].cpu(), num_samples=1,
                                              generator=sample_generator).item()

          #（6）抽到'.'（编号是0）说明名字结束了：先判断再append，所以打印的名字不带'.'
          if character_index == 0:
            break

          #（7）记录字符，并把上下文滑动一格
          generated_characters.append(itos[character_index])
          context = context[1:] + [character_index]

        #（8）拼成字符串就是生成出来的一个名字
        print(''.join(generated_characters))
  finally:
    #（9）不管生成过程有没有出错，都把BatchNorm模式还原，不影响调用者
    restore_batch_norm_modes(saved_modes)


#=========================== 交互式采样：输入个数就能生成名字 ===========================
#（1）训练和评估都做完以后，进入一个交互循环：
#   输入一个数字，就生成这么多个名字；输入no，就结束程序；直接回车则用默认个数
#   （默认个数由命令行参数--number_of_names决定，默认是10）
def ask_and_generate(default_number_of_names):
  while True:
    user_input = input(f'请输入要生成的名字个数（回车=默认{default_number_of_names}，输入 no 结束程序）: ').strip()

    #（2）输入no（不区分大小写，前面strip过，所以多余空格也被去掉了）就结束程序
    if user_input.lower() == 'no':
      print('程序结束')
      return

    #（3）直接回车：用命令行参数给的默认个数
    if user_input == '':
      number_of_names = default_number_of_names

    #（4）输入了数字：按用户要的个数生成
    elif user_input.isdigit():
      number_of_names = int(user_input)

    #（5）其他输入：提示一下，然后重新问
    else:
      print(f'无法识别"{user_input}"，请输入一个正整数，或者输入 no 结束程序')
      continue

    if number_of_names <= 0:
      print('个数要大于0，请重新输入')
      continue

    print(f'\n生成{number_of_names}个名字：')
    use(number_of_names)


#（6）开始交互式采样
#   用Ctrl+C也能强制退出：这里捕获KeyboardInterrupt，避免打出一大段报错信息
try:
  ask_and_generate(arguments.number_of_names)
except KeyboardInterrupt:
  print('\n收到 Ctrl+C，程序结束')
