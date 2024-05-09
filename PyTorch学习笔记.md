神经网络本质为矩阵乘法，通过乘法对输入矩阵降维处理，只是给每层网络套了一层激活函数，实现高维度数据-->低维度，故又可称为TensorFlow

当神经网络输出不稳定时检查每层网络梯度的模，查看梯度是否有出现过大或过小，是否有小于0或大于20

模型复杂度应与实际模型复杂度相符合，否则会出现Underfitting与Overfitting，实际中更容易overfitting。通过增加层数与神经元数量可以增加复杂度。


---

## 回归问题（梯度下降）
linear regression、 logistic regression、classification（每种类别概率之和为1）  

$x^*=x-\alpha \frac{\partial loss}{\partial x}$  
取值：$0.0001<\alpha<0.01$  

梯度下降求解函数：sgd、rmsprop、adam、momentum、nag、adagrad、adadelta  

回归问题中关于拟合偏差的理解可以看作是noise（高斯噪声）的影响。 $ y = a*x+b+noise $  

**对数据做线性回归示例见LinearRegression\.py，训练数据见training data LinearRegression.csv**  

---

## 分类问题  
**手写数字识别，(神经网络)**  
参考资料：
https://blog.csdn.net/v_JULY_v/article/details/51812459?spm=1001.2014.3001.5506  

![神经网络实现过程图1](image/000017神经网络实现过程图1.png "神经网络实现过程图")  
![神经网络实现过程图2](image/000017神经网络实现过程图2.png "神经网络实现过程图")  
![神经网络实现过程图3](image/000017神经网络实现过程图3.png "神经网络实现过程图")
训练集为28x28px图像，位图照片每个px取值为0/1（图片为28x28的0/1矩阵），将28维方阵变为一维矩阵 $[....]_{1*784}$ 。 

下图为表示目标函数模型， $h_x = w*x+b，H_3表示识别的数字$  
![手写数字识别](image/003000pt手写数字识别1.png "手写数字识别算法1")
输出表示$H_3=[...]_{1*10}；e.g.1=>[0,1,0,0,0,0,0,0,0,0]；e.g.3=>[0,0,0,1,0,0,0,0,0,0]$，该表示方法称为one_hot  

在此HiddenLayer激活函数不使用sigmoid，使用ReLU作为激活函数，该函数当x<0时输出y=0；x>0时y=x；导数y'==1。下图为加上ReLU激活函数后的神经网络模型。  
![ReLU函数图像](image/003001pt激活函数relu图像.png)  
使用右函数求出输出预测值pred矩阵中最大概率的标号，即可得到最终预测的数字。**argmax(pred)**  
**一般而言最后一层输出网络的激活函数不使用ReLU函数，可以根据实际场景选择使用sigmoid等**  


code见NeuralNetworksNUMIdentify\.py，numi手写字母数据集见mnist_data文件夹

---
<div STYLE="page-break-after: always;"></div>


# 一、pytorch基本数据类型及运算  
### 1. Tensor声明
IntTensor(int标量、int向量)、FloatTensor(float标量、float向量)，没有string类型。当需要表示string类型时怎么表示？————使用**one-hot编码**方法表示，将类型结果编码成一维矩阵。但是使用one-hot编码的弊端是表示的信息量太小，Word2vec、glove编码器可以解决此问题。  
![数据类型](image/003002pt数据类型.png)  

>向量：
>\> a = torch.tensor([1.1])    &emsp;&emsp;#生成一个一维向量[1.1]
>\> a = torch.randn(2, 3)  &emsp;&emsp;#生成一个2*3维数值服从N(0, 1)的随机数组
>\> a = torch.rand(1,2,3)   &emsp;&emsp;#将生成一个元素在(0,1)之间的随机数，dim=3的size=[1,2,3]矩阵；此时a[0]={第0维度的2x3矩阵}
>\> a.type()   &emsp;&emsp;&emsp;&emsp;#return：a的数据类型
>\> isinstance(a, torch.FloatTensor)   &emsp;&emsp;#数据类型检验，当a为torch.FloatTensor类型时返回true
>
>标量：
>\> torch.tensor(1.)   &emsp;&emsp;#return num：1.
>\> torch.tensor(1.3)  &emsp;&emsp;#return num：1.300
>
>数据大小：
>\> a.shape    &emsp;&emsp;#标量return: rotch.size([])，向量return: rotch.size([row, column])
>\> a.shape(2) 或 a.size(2) &emsp;&emsp;#当a.shap为[2,3,5]时将返回shape的第2位元素即为5
>\> a.size()   &emsp;&emsp;&emsp;&emsp;#同a.shape
>\> len(a.shape)   &emsp;&emsp;#标量return: 0，向量return: row num
>\> a.dim()     &emsp;&emsp;&emsp;&emsp;#return a的rank
>\> a.numel()   &emsp;&emsp;#return：tensor占用的大小。如a.shap为[2,3,5]则a.numel()为 2\*3\*5
>
>
>
>创建向量：
>\> torch.tensor([1.1])
>\> torch.FloatTensor(2)   &emsp;&emsp;#生成一个1行2列的服从N(0, 1)随机分布的向量
>
>\> data = np.ones(2)    &emsp;&emsp;#调用numpy生成一个矩阵[1, 1]，当为np.ones([2,3])将生成一个2x3维向量
>\> torch.from_numpy(data)     &emsp;&emsp;#从numpy引入data，此时data是一个float64数据
>
>\> 1维向量 Dimension 1/rank 1：
>bias和linear input中使用得多，loss使用标量即[]不使用向量
>**关于矩阵的dim即rank以及shape/size的理解：dim/rank表示矩阵的维数，例如二维矩阵的dim=2，三维矩阵ones(2,3,6)的dim=3。shape/size表示矩阵是 几x几 的矩阵，例如ones(2,2,3)则shape=size=[2,2,3]**
>
>dim3：适用于RNN文字处理
>dim4：适用于CNN图片处理，[照片张数b, 通道数c, 像素高度h, 像素宽度w]
>

### 2. Tensor创建  
* **在np中创建后导入Tensor**  
>\> a = np.array([2,3.3])
>\> torch.from(a)
>
>\> torch.tensor([2., 3.2])     &emsp;&emsp;#小写tensor入口参数为数字(立即数)
>\> torch.Tensor(1, 3, 6)     &emsp;&emsp;#大写Tensor的.type()使用pytorch的默认数据类型，一般通过**torch.set_default_tensor_type(torch.DoubleTensor)**更改默认type()为double类型、FolatTensor入口参数为矩阵的size()
>\> torch.Tensor([2., 3.2])     &emsp;&emsp;#大写Tensor、FloatTensor入口参数也可以是立即数，但是需要用[ ]框起来
>\> torch.empty(2,3)        &emsp;&emsp;#申请一个2x3矩阵的内存，此时矩阵内部数字没赋值数值不确定
>
* **随机初始化**
>\> a = torch.rand(3,3)     &emsp;&emsp;#随机[0, 1]分布
>\> a = torch.randn(3,3)    &emsp;&emsp;#N(0, 1)分布
>\> a = torch.full([2, 3], 6)   &emsp;&emsp;#生成一个数值全为6的2x3矩阵，入口参数(shape, tensor值)；生成一个标量的入口([], 6)
>\> b = torch.rand_like(a)      &emsp;&emsp;#_like(a)即将a的shape传给rand()
>\> b = torch.randperm(10)      &emsp;&emsp;#随机数种子，数值在[0, 10)step=1，但顺序打散，如b=[1,5,3,2,8,0,9,4,7,6]
>\> c = torch.randint(min, max, [shape])    &emsp;&emsp;#入口参数1,2为数值取值的[min, max)，第三个参数为tensor的shape如[2,3]
>\> d = torch.arange(start, end, step)    &emsp;&emsp;#生成一个[start, end)的等差数列，如入口(0, 11, 2)。当arange(8)则为[8]
>\> d = torch.linspace(start, end, steps=num)   &emsp;&emsp;#生成[start, end]且有等距切割成num个元素的等差数列，如入口(0,10, steps=2)
>\> d = torch.logspace(start, end, steps=num)   &emsp;&emsp;#生成[10^start, 10^end]之间等差分布的num个元素数列，如入口(0, 0.1, steps=10)
>\> d = torch.ones(3, 3)    &emsp;&emsp;#3x3的全1矩阵
>\> d = torch.ones_like(a)
>\> d = torch.zeros(3, 4)   &emsp;&emsp;#3x4全0
>\> d = torch.eye(3, 4)     &emsp;&emsp;#3x4对角线为1

* **randperm()用途**
>\> a =torch.rand(4,3)
>\> idx = torch.randperm(4)
>\> a[idx]     &emsp;&emsp;#将矩阵a的行向量顺序按照idx的顺序打乱重排，即a的序号按照idx做索引

### 3. tensor的索引、切片
>\> a = torch.rand(4, 3, 28, 28)     &emsp;&emsp;#a为4张照片rgb通道28*28px的照片
>\> a[0]、a[0, 0]、a[0, 0, 2, 4]    &emsp;&emsp;#结果为第0张照片全部内容、第0张照片r通道的所有px、0张r通道px[2, 4]
>\> a[:2]、a[:2, 1:, :, :]、a[:-1]      &emsp;&emsp;#结果为0-1张照片、0-1张照片的1-2通道的所有px
>\> a[:, :, ::2, :21:2]     &emsp;&emsp;#所有照片所有通道的隔2行采一次px，0-20列隔2个采一次px
>\> a.index_select(1, torch.tensor[0, 2])       &emsp;&emsp;#对a的第1个序号即a[:, 0:3]索引0,2序号([]为tensor型)，即取a[:, 0],a[:, 2]
>\> a[...]、a[0, ...]、a[0, ..., :2]        &emsp;&emsp;#...表示取所有，a[0, ..., :2]0张照片的所有rgb,行px的0,1列px
>
>\> a = torch.randn(3, 4)
>\> mask = a\.ge(0.5)    &emsp;&emsp;#将a中大于0.5的元素置1，其余置0
>\> torch.masked_select(a, mask)    &emsp;&emsp;#取出a中对应mask元素为1的位置的元素，即将a中概率大于0.5的数取出来，并按位置顺序排成一行
>\> torch.take(a, torch.tensor([0, 2, 8]))    &emsp;&emsp;#先将a矩阵打平成一行，然后选取第0,2,8位数据

### 4. 维度变换：view/reshape、squeeze/unsqueeze、transpose/t/.permute、expand/repeat
* **view/reshape 功能一样**
>\> a = torch.rand(4, 3, 28, 28)
>\> b = a[:, 0]
>\> b.view(4, 28*28)    &emsp;&emsp;#将b转换为4,28x28的矩阵，即将b的一张照片通道和px合并后打平成1行矩阵

* **squeeze/unsqueeze：压缩/增加dim**
>\> b.unsqueeze(0)      &emsp;&emsp;#在b的第dim 0维度前增加一个维度，它的shape为[1, 4, 1, 28, 28]
>\> b.unsqueeze(-2)     &emsp;&emsp;#在b的第dim -2维度后增加一个维度，它的shape为[1, 4, 1, 28, 1, 28]
>\> b.squeeze(-2)       &emsp;&emsp;#挤压b第dim -2维，即shape由[1, 4, 1, 28, 1, 28]变为[1, 4, 1, 28, 28]
>\> b.squeeze()         &emsp;&emsp;#挤压掉b的所有只有1个元素的dim，即shape由[1, 4, 1, 28, 28]变为[4, 28, 28]

* **expand(只是扩展shape,不复制数据)/repead(增加了数据，不建议)：扩展某一dim的shape**
>\> a = b = torch.rand(1, 1, 32, 32)
>\> b.expand(4, 3, 32, -1)      &emsp;&emsp;#扩展之后shape为[4, 3, 32, 32]，**-1表示当前维度shape不变**
>\> a.repeat(4, 3, 1, 32)        &emsp;&emsp;#参数为将每个dim上数据重复的次数，a的shaper由[1, 1,32,32]变为[4, 3, 32, 32*32]
>\> c.expand_as(a)      &emsp;&emsp;#将c矩阵扩展成与a同型的

* **transpose/t/.permute**
>\> a = torch.rand(4, 3, 28, 28)
>\> b = torch.tensor([\[1, 2], [3, 4]])   &emsp;&emsp;#2dim矩阵
>\> b.t()       &emsp;&emsp;#b转置，.t()只能用于2dim的矩阵
>\> a.transpose(1, 3)   &emsp;&emsp;#交换a的dim1与dim3，即shape由[4, 3,28,28]变为[4, 28, 28, 3]

**transpose之后数据在内存中变得不连续了，此时使用.contiguous()将数据变成连续的之后再使用view()**

>\> a1 = a.transpose(1, 3).contiguous().view(4, 3\*32\*32).view(4,32,32,3).transpose(1, 3)      &emsp;&emsp;#a又还原成了原来的a
>
>\> torch.all(torch.eq(a, a1))      &emsp;&emsp;#矩阵相等比较：eq对a,a1进行比较，all当全部相等时返回true
>
>\> a = torch.rand(4, 3, 28, 20)
>\> a.permute(0, 2, 3, 1)       &emsp;&emsp;#输入为dim的顺序号重排，shape由[4, 3,28,20]变为[4, 28, 20, 3]


### 5. broadcasting 根据两个矩阵的样式自动扩展成可以相加的矩阵，扩展时对缺省的矩阵行列进行复制操作。
**说明：broadcasting是一种运算机制，不需要使用特定代码指定，直接A+B即可**
![broadcasting计算过程](image/003003pt函数broadcasting.png)
使用场景：
dim(A)>dim(B)，需要A与B相加，此时增加上B缺省的维度使dim(B)==dim(A)
* 1。 A+B：其中A.shape为[4, 32,14,14]，B.shape为[1, 32,1,1]。此时自动对B扩张为[4, 32,14,14]。**对于shape为1的dim自动扩展shape**
* 2。 A+B：其中A.shape为[4, 32,14,14]，B.shape为[14, 14]。此时自动对B扩张为[4, 32,14,14]。**对于没给的dim自动扩充，默认从低位dim开始匹配**
* 3。 A+B：其中A.shape为[4, 32,14,14]，B.shape为[2, 32,14,14]。此时不能直接使用broadcasting。
>学生分数普遍比较低，给每个学生分数加5分。其中成绩：A=[class, students, scores]，A.shape为[4, 32, 8]，加分B=tensor[5.]，B.shape=[1]。当使用broadcasting时，B自动复制扩充B.shape为[4, 32, 8]其中B[0, 0, :]=[5, 5, 5, 5, 5, 5, 5, 5]
>\> A += B  &emsp;&emsp;#即可得到结果

### 6. 拼接与拆分cat/stack/split(按长度拆)/chunk(数量)
* **cat拼接，直接两个tensor相加拼接**
>\> a = torch.rand(4, 32, 8)
>\> b = torch.rand(5, 32, 8)
>\> torch.cat([a, b], dim=0)    &emsp;&emsp;#将a,b在dim=0维度进行合并，得到的shape为[9, 32, 8]。dim=1为行dim=2为列
* **stack，会创建新的维度dim，将两个shape相同的矩阵后面的维度放到两个平行维度dim**
>\> a = b = torch.rand(4, 3, 16, 32)
>\> torch.stack([a, b], dim=2)      &emsp;&emsp;#在dim=2前插入一个维度，得到的shape为[4, 3, 2, 16, 32]，stack将a,b的第2,3维dim映射到两个不同空间

问题：1班有32个人每个人8个成绩记a=tensor.rand(32,8)，2班有32个人每个人8个成绩记b=tensor.rand(32,8)，将两个班级成绩合并到c
>\> c = torch.stack([a, b], dim=0)      &emsp;&emsp;#得到的c.shape为[2, 32, 8]
* **split，根据拆分后矩阵的长度拆分**
>\> c = torch.rand(3, 32, 8)
>\> a, b = c.split([2, 1], dim=0)   &emsp;&emsp;#按长度拆分。将dim=0维度分成2、1维度,a.shape\==[2, 32,8],b.shape\==[1, 32,8]
>\> a, b, d = c.split(1, dim=0)        &emsp;&emsp;#按长度拆分。将dim=0维度拆分成每长度均为1,a.shape\==b\==d\==[1, 32,8]
* **chunk，根据拆分后矩阵的数量拆分**
>\> c = torch.rand(4, 32, 8)
>\> a, b = c.chunk(2, dim=0)    &emsp;&emsp;#在dim=0位将c拆分成2个矩阵。a.shape\==b.shape\==[2, 32, 8]

### 7. 数学运算 
>\> a = torch.rand(4,4)
>\> b = torch.rand(4)
>\> torch.add(a, b) <\==> a + b   &emsp;&emsp;#b自动补为shape为[3, 4]
>\> torch.sub(a, b) <\==> a - b
>\> torch.mul(a, b) <\==> a * b    &emsp;&emsp;#该方法为矩阵对应位置相乘，即matlab中.*运算
>\> torch.div(a, b) <\==> a / b
>
>\> c = torch.full([2, 2], 3)
>\> c.pow(2) <\==> c\*\*2    &emsp;&emsp;#对c每个位置进行平方运算
>\> c.sqrt() <\==> c\*\*(0.5)        &emsp;&emsp;#对c每个位置元素开平方根
>\> c.rsqrt() <\==> c\*\*(-1)       &emsp;&emsp;#对c每个位置元素求倒数即1/3
>\> torch.exp(c)    &emsp;&emsp;#对c每个位置元素求e^3
>\> torch.log(c)    &emsp;&emsp;#对c每个位置元素求ln3
>\> torch.log2(c)   &emsp;&emsp;#对c每个位置元素求 $log_2 3$

**dim=2 矩阵乘法**
>\> a = b= torch.rand(4, 4)
>\> torch\.mm(a, b)   &emsp;&emsp;#.mm() 只适用于dim=2的矩阵相乘
>\> torch.matmul(a, b) <\==> a @ b     &emsp;&emsp;#矩阵相乘

**dim>=3 矩阵乘法**
>\> a = torch.rand(4, 3, 28, 64)
>\> b = torch.rand(4, 3, 64, 32)
>\> torch\.mm(a, b) <\==> torch.matmul(a, b)    &emsp;&emsp;#只对矩阵最后两个维度进行相乘，结果的shape为[4, 3, 28,32]
>\> c = torch.rand(4, 1, 64, 12)
>\> torch\.mm(a, c)      &emsp;&emsp;#此时符合broadcasting机制，计算结果的shape为[4, 3, 28, 12]

**近似值：floor、ceil、round、trunc、frac**
>\> a = torch.tensor(3.14)
>\> a.floor()、a.ceil()、a.trunc()、a.frac()    &emsp;&emsp;#小于a的最大整数、大于a的最大整数、a的整数部分、a的小数部分
>\> a.round()   &emsp;&emsp;&emsp;&emsp;#对a四舍五入

**梯度裁剪clamp：**
>\> 假设grad=torch.rand(2,3)*15
>\> grad.max()  &emsp;&emsp;#返回grad中最大的元素
>\> grad.median()   &emsp;&emsp;#返回grad中的中位数
>\> grad.clamp(10)      &emsp;&emsp;#目的是将grad中小于10的元素值改为10。给出的是最小值
>\> grad.clamp(0, 10) <==> grad = torch.clamp(grad, 0, 10)   &emsp;&emsp;#将grad的值限制在[0 , 10]的范围

### 8. 统计属性norm、mean/sum、prod、max/min/argmin/argmax、kthvalue/topk
* **norm：求范数，向量的模**
    >给定一个向量 $\mathbf{x}$，它的 $p$ 范数(norm)（$L_p$ 范数）定义为：
    >$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$$
    >其中 $n$ 是向量的维度，$x_i$ 是向量的第 $i$ 个元素。当 $p=1$ 时，这就是 $L_1$ 范数，也称为曼哈顿范数；当 $p=2$ 时，这就是 $L_2$ 范数，也称为欧几里得范数；当 $p=\infty$ 时，这就是 $L_{\infty}$ 范数，也称为最大范数或者无穷范数。

>\> a = torch.full([2, 2, 2], 1)
>\> a.norm(p)   &emsp;&emsp;#a矩阵的p范数，将a的所有元素绝对值的一次方相加后开1次方
>\> a.norm(1, dim=1)    &emsp;&emsp;#在a的dim=1维度上求1范数，左方的范数的dim为1，shape为[2 ]，内容为=[4, 4]
>\> a.norm(2, dim=2)    &emsp;&emsp;#在a的dim=2维度求2范数，左方结果为dim=2，shape为[2, 2]，内容为 = [\[1.141, 1.141], 下一行[1.141, 1.141]]
* **mean/sum、prod、max/min/argmin/argmax**
>\> a = torch.rand(2, 10)
>\> a.min()、a.max()、a.mean()、a.prod()、a.sum()    &emsp;&emsp;#矩阵a所有元素的最小值、最大值、平均值、累乘(连乘)、求和
>\> a.min(dim=1)    &emsp;&emsp;#返回参数为元组 ( [ 最小值的值 ], [ 最小值的序号 ] )
>\> a.argmax()、a.argmin()      &emsp;&emsp;#先将a打平成一行矩阵，返回a元素最大值最小值所在的序号
>\> a.argmax(dim=1)、a.argmin(dim=1)    &emsp;&emsp;#求a每一个dim=0维度的shape上，dim=1维度内求最大/小值编号，左式dim为1，shape为[2]，结果为 = [2, 6]表示在第一张照片上最大值编号为2,第二站编号为6
>\> a.max(dim=1, keepdim=True)      &emsp;&emsp;#dim=1表示在a的dim=1维度求max，得到矩阵为1行4列，keepdim=True将得到的矩阵转置为4行1列的矩阵。
* **topk(n)：返回n个概率最大的值(从大到小排)并返回对应的序号，kthvalue(n)：返回第n小的值且只能求从小到大排序**
>\> a = torch.rank(4, 10)
>\> a.topk(3, dim=1)    &emsp;&emsp;#在a的dim=1维度上计算数值最大的3个数并返回数字和序号，计算结果：2个dim为2的矩阵拼成的元组，shape为$([4, 3]_{max数值}, [4, 3]_{标号})$
>\> a.topk(3, dim=1, False)     &emsp;&emsp;#返回dim=1维度上概率最小的3个值和序号
>\> a.kthvalue(5, dim=1)        &emsp;&emsp;#返回dim=1维度上概率第5小的值和序号
### 9. 比较>、>=、<、<=、!=、==、eq() 
>\> a = torch.rank(4, 10)
>\> a>0.5       &emsp;&emsp;&emsp;&emsp;#返回一个与a同型的矩阵，也是按位比较，成立则该位为1否则为0
>\> eq(a, a)    &emsp;&emsp;&emsp;&emsp;#只有当两个矩阵相等时才返回True，否则False


---
<div STYLE="page-break-after: always;"></div>

# 二、Pytorch高阶操作：where(映射)、gather(映射分类)
### 1. torch.where(condition, x, y) --> Tensor，此处可以使用GPU并行运算
![where函数功能示意图](image/003004pt函数where.png)

### 2. torch.gather(input, dim, index, out=None) --> Tensor
![gather函数功能示意图](image/003005pt函数gather.png)
![gather函数实际使用代码示意图](image/003006pt函数gather示例.png)


<div STYLE="page-break-after: always;"></div>


# 三、梯度(函数值在某方向增加最快的方向)
随机初始化 [ θ ]后，在计算中先求梯度再计算 $\theta_i=\theta_i-\alpha*grad(Loss_{\theta})$  

**局部最小值：** 当神经网络层数变多以后cost function会有很多局部极小值，此时的难题便是如何跳出局部极小值从而找到全局最小值  
**鞍点：** 鞍点处的梯度均为0
![鞍点图像](image/003007鞍点图像.png)
正确找到全局最小值的影响因素：优化器选择、初始化值、学习率、逃离局部最小值的动力

### 1. sigmoid函数：连续光滑且可导，导数与函数值均有限，
缺点是：当x很大时他的导数≈0，会导致梯度长时间得不到更新
$$f_{(x)} =\sigma_{(x)} = \frac{1}{1 + e^{-x}}$$
$$ \sigma' = \sigma * (1 - \sigma) = \sigma_{(x)} - \sigma_{(x)}^2 $$
>\> a = torch.linspace(-100, 100, 10)
>\> torch.sigmoid(a)    &emsp;&emsp;#对矩阵a的每个元素套上sigmoid

### 2. tanh函数：
$$f_{(x)} = tanh_{(x)} = \frac{(e^x - e^{-x})}{(e^x + e^{-x})} = 2 * sigmoid(2x) - 1$$
$$tanh_{(x)}' = 1 - tanh_{(x)}^2$$
>\> torch.tanh(a)

### 3. RELU函数：适合做深度学习的激活函数
当x<=0时，y=0；x>0时，y=x。RELU函数导数很简单，当x>=0时，导数恒为1。

$$ f_{(x)}=\begin{cases}
0 & (x<0) \\
x & (x>=0) 
\end{cases}$$

>\> torch.relu(a)

### 4. costfunction及梯度
计算loss时注意没有开根号！因次使用L2-norm范数不合适。
loss = norm $(y - (w*x + b))^2$
>\> loss = torch.norm(y - (w*x+b), 2).pow(2)

对平方求导：
**Way1**
>\> x = torch.ones(1)
>\> w = torch.full([1], 2)
>\> w.requires_grad_()      &emsp;&emsp;#声明w参数需要梯度运算，声明之后才能对w求导。或者在初始化w时声明： **w = torch.tensor([2.], requires_grad_()=True)**
>\> mse = F.mse_loss(x\*w, torch.ones(1))    &emsp;&emsp;#计算loss是标量。入口参数：( 预测值，实际值 )。计算： $((x * w)-1)^2 = (2-1)^2$
>\> torch.autograd.grad(mse, [w])       &emsp;&emsp;#计算loss的梯度。注意流程：声明w需要梯度信息，->求loss，->求loss的梯度。计算 $\frac{\alpha Loss}{\alpha w}$

**Way2**
>\> 同上
>\> mse = F.mse_loss(x\*w, torch.ones(1))
>\> mse.backward()      &emsp;&emsp;#此时不会返回tensor，但是w的信息存放在w.grad矩阵中。查看某个位置的梯度时w.grad[0]

**对softmax激活函数求导**
**==softmax常与MSE连用！！==**
softmax函数可以保证神经网络输出节点的概率之和为1，并求出每一个输出的概率。经过softmax函数后可以将softmax之前的数据的数值差距拉开的更明显。 
![softmax函数示意图](image/003008pt函数softmax.png)
![softmax求导计算过程](image/003009pt函数softmax求导1.png)
![softmax求导计算过程](image/003009pt函数softmax求导2.png)
$p_i为softmax函数的输出$

$$ \frac{\alpha p_i}{\alpha a_j}=\begin{cases}
p_i(1 - p_j)=p_i(1-p_i) & (i=j) \\
-p_i*p_j & (i≠j)
\end{cases}$$

>\> a = torch.rand(3)
>\> a.requires_grad_()
>\> p = F.softmax(a, dim=0)
>\> p.backward(retain_graph=True)   &emsp;&emsp;#当p.backward()报错时加上这句话
或使用这个函数：
>\> p0 = torch.autograd.grad(p[0], [a], retain_graph=True)
>\> p1 = torch.autograd.grad(p[1], [a])
>\> p2 = torch.autograd.grad(p[2], [a])     &emsp;&emsp;#注意在使用torch.autograd.grad()函数时，loss必须是shape为[1]的矩阵或者标量才行。左侧结果为p2对a0、a1、a2的梯度。

### 5. Stochastic Gradient Descent（随机梯度下降）
见八、5节内容

---

<div STYLE="page-break-after: always;"></div>


# 四、感知机（神经网络）
### 1. 单层感知机
一种简单的神经网络，可完成二元线性分类问题，但是引入干扰后难以正确识别。bias偏置改变了决策边界的位置
![感知机模型](image/003012pt感知机.png)
$神经元x_j^{(i)}表示第i层第j个节点，权重w_{jk}^{(i)}$表示第i层第k个元素中上一层的第j个节点在加权计算中的权重
$$\begin{cases}
pred =sigmoid( \sum{(x_i*w_i)}+b)=\sigma(z_{(x)})\\
loss = \frac{1}{2}(O_0^1-t)^2
\end{cases}$$
**此时的loss称为MSE(mean squid error)**

![单层NN对loss求偏](image/003013pt单层NN的loss求导.png)
$$\frac{\alpha E}{\alpha  w_{j0}} = (O_0-t)O_0(1-O_0)x_j^{(0)}$$

>\> x = torch.randn(1, 10)
>\> w = torch.randn(1, 10, requires_grad=True)
>\> o = torch.sigmoid(x@w.t())
>\> loss = torch.mse_loss(o, torch.ones(1,1))
>\> loss.backward()     &emsp;&emsp;#得到了loss关于w的偏导数w.grad

### 2. 多层感知机的一个隐藏层
![多层NN对loss求偏导](image/003014pt多层NN的loss偏导.png)
$$\frac{\alpha E}{\alpha  w_{jk}} = (O_k-t_k)O_k(1-O_k)x_j^{(0)} $$
**使用链式求导将每层神经网络级联起来**

>\> x = torch.randn(1, 10)
>\> w = torch.randn(2, 10, requires_grad=True)
>\> o = torch.sigmoid(x@w.t())
>\> loss = F.mse_loss(o, torch.ones(1,2))
>\> loss.backward()     &emsp;&emsp;#loss对w的梯度保存在w.grad数组中，它的shape为[2, 10]

### 3. 链式法则怎么写
![网络图](image/003015pt链式求导示例网络.png)

>\> x = torch.tensor(1.)
>\> w1 = torch.tensor(2., requires_grad=Ture)
>\> b1 = torch.tensor(1.)
>\> w2 = torch.tensor(2., requires_grad=Ture)
>\> b2 = torch.tensor(1.)
>\> y1 = w1*x+b1
>\> y2 = w2*y1+b2
>\> dy2_dy1 = autograd.grad(y2, [y1], retain_graph=True)
>\> dy1_dw1 = autograd.grad(y1, [w1], retain_graph=True)    &emsp;&emsp;#链式法则求导dy2/dw1
>\> dy2_dw1 = autograd.grad(y2, [w1], retain_graph=True)    &emsp;&emsp;#直接调用函数求dy2/dw1

### 4. 多层感知机的反向传播
![神经网络的链式求导即反向传播推导](image/003016pt神经网络链式求偏导推导.png)
上图最终表达式为： $\frac{\alpha E}{\alpha W_{ij}} = O_j(1-O_j)O_i*\sum\limits_{k∈K}{{(O_k-t_k)O_k(1-O_k)W_{jk}}} $
$$可记作：\frac{\alpha E}{\alpha W_{ij}} = O_j(1-O_j)O_i\sum\limits_{k∈K}{{\delta _k^{(k)} W_{jk}}}$$
$\delta _i^{(k)}=(O_i^{(k)}-t_i^{(k)})O_i^{(k)}(1-O_i^{(k)})$记为第k层第i个元素的梯度信息，对于任意第J层的δ怎么算：(其中K为J的下一层即K=j+1)
$$\delta_i^{(J)} = O_i^{(J)}(1-O_i^{(J)})\sum\limits_{k∈K} \delta _k^{(K)} W_{jk}^{(K)} $$

### 5. 2D(二元)函数优化实例
Himmelblau function函数用来验验证模型是否可以找到全局最小值，其中图像蓝色部分表示局部极小值点且四个局部最小值均为0。
![模型效果验证函数Himmelblau](image/003017pt模型验证函数Himmelblau.png)

$$\begin{cases}
f_{(3.2, 2.0)}=0.0\\
f_{(-2.805118,3.131312)}=0.0\\
f_{(-3.779310,-3.283186)}=0.0\\
f_{(3.584428,-1.858126)}=0.0
\end{cases}$$

code见文件：NN GradientDescent in HimmelblauFunction\.py

### 6. Classification 示例（交叉熵）
使用神经网络给 y = w*x +b 套一层 激活 函数
激活函数使用**cross entropy**函数，它的loss称为cross entropy loss，将会惩罚小于0的结果
$$entropy = -H_{(P_{(i)})} = -\sum\limits_i P_{(i)}log_2P_{(i)} = \sum\limits_i P_{(i)}log_2\frac{1}{P_{(i)}} $$当entropy值越大则系统越稳定，值越小混乱程度越大即越不稳定。

**Cross Entropy：**
网络中Cross Entropy Loss由两个模块组成：softmax --> log
**参考文章：https://blog.csdn.net/tsyccnh/article/details/79163834**

交叉熵表示信息量，其中当越不可能的事件发生了，我们获取到的信息量就越大；越可能发生的事件发生了，我们获取到的信息量就越小。
$$Cross Entropy：H_{(p,q)} = -\sum P_{(x)}log_2q_{(x)} $$ 
$log_2q_{(x)}$定义为事件$X=x_0$的信息量，其中$q_{(x)}$为该事件发生的概率，下图为信息量的函数图，由此可见事件发生概率越小它的值越大，即loss越大  
![信息量函数图](image/003018pt熵信息量图.png)

>**补充**：关于实际使用中交叉熵的理解：
>**相对熵（KL散度）** 相对熵又称KL散度,如果我们对于同一个随机变量 x 有两个单独的概率分布 P(x) 和 Q(x)，我们可以使用 KL 散度 来衡量这两个分布的差异。
>在机器学习中，**P往往用来表示样本的真实分布**，比如[1,0,0]表示当前样本属于第一类。**Q用来表示模型所预测的分布**，比如[0.7,0.2,0.1]
>$$DKL(p||q)=\sum\limits_{i=1}^n p_{(x_i)}log_2\frac{p_{(xi)}}{q_{(xi)}}$$
>其中：n为事件的所有可能性。DKL 的值越小，表示q分布和p分布越接近。上式相对熵展开为
>$$DKL(p||q)=\sum\limits_{i=1}^n p_{(x_i)}log_2 p(x_i)−\sum\limits_{i=1}^n p(x_i)log_2q_{(x_i)} \\ 
>= −H_{(p_{(x)})} − \sum\limits_{i=1}^np_{(x_i)}log_2q_{(x_i)} $$

取**交叉熵作为loss：梯度下降的更快** $$loss = -\sum\limits_{i=1}^n y_i log_2\hat{y_i} $$
![交叉熵实际应用中计算](image/003019pt交叉熵实际应用举例.png)

>\> x = torch.randn(1, 784)
>\> y = torch.randn(10, 784)
>\> logits = x@y.t()
>\> pred = F.softmax(logits, dim=1)     &emsp;&emsp;#softmax函数求出每个pred的概率
>\> pred_log = torch.log(pred)
>\> loss = F.cross_entropy(logits, troch.tensor([3]))
>\> loss = F.nll.loss(pred_log, troch.tensor([3]))     &emsp;&emsp;#如果不使用.cross_entropy()函数的话可以用这个，推荐使用.cross_entropy()

* 多分类问题的神经网络结构图如下，3层网络，784个input、2个hiddenlayer、10个output。
![多分类问题的神经网络结构图](image/003020pt多分类问题网络图.png)

code见文件 NN Classification Example\.py

---

<div STYLE="page-break-after: always;"></div>


# 五、常用激活函数
### 1. Tanh、Sigmoid（可能会有梯度离散）
![Tanh和Sigmoid](image/003021Tanh和Sigmoid.png)

### 2. ReLU（一定程度解决了tanh、sigmoid的梯度离散）
![ReLU](image/003022ReLU.png)

### 3. Leaky ReLU（ReLU改进版，缓解了梯度离散）
![Leaky ReLU](image/003023LeakyReLU.png)
可以修改x<0的斜率

### 4. SELU（解决了0点不连续）
![SELU](image/003024SELU.png)

### 5. softplus（在0附近梯度均匀变化）
![softplus](image/003025softplus.png)

---

<div STYLE="page-break-after: always;"></div>


# 六、使用GPU计算(.to(device)将模块和数据搬去GPU)

device = torch.device('cuda:0')
net = MLP().to(device)
data, target = data\.to(device), target\.to(device)

---

<div STYLE="page-break-after: always;"></div>


# 七、验证集
在test data上还需要求解 Loss 与 Accuracy 
使用 .argmax()、.eq()、sum() 函数得到正确输出的数量，之后除以测试数量得到Accuracy

## 1. visdom可视化
visdom服务下载与开启：
* 下载：pip install visdom
* 卸载: pip uninstall visdom
* 开启：python -m visdom.server

**单条曲线：**
from visdom import Visdom

viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))       &emsp;&emsp;#入口（y轴坐标，x轴坐标，图像窗口ID，图像窗口显示的标题），坐标值只接收.cpu.np数据
viz.line([loss.item()], [global_step], win='train_loss', updata='append')   &emsp;&emsp;#入口（y坐标，x坐标，图像窗口ID，添加显示的数据）

**多条曲线：**
from visdom import Visdom

viz = Visdom()
viz.line([\[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))    &emsp;&emsp;#legend=['loss', 'acc.']表示y1，y2的图标
viz.line([\[test_loss, corrent / len(test_loader.dataset)]], [global_step], win='test', updata='append')

**显示图像**
from visdom import Visdom

viz = Visdom()
viz.images(data.view(-1, 1, 28, 28), win='x')
viz.text(str(pred.detach().cpu.numpy()), win='pred', opts=dict(title='pred'))

---




# 八、Underfitting And Overfitting
train set、val set、test set。 val set交叉验证集是为了挑选更好的模型，test set是客户用来验证模型的数据集。
关于如何找到最好的参数？ 使用使val set拥有最小error的那一组参数。使用k-fold方法划分train set 和val set，即从train set轮流划分出来一部分数据作val set

**Occam's Razor剃须刀原理：More things should not be used than are necessary**

防止Overfitting：
* 更多训练数据
* 减小模型复杂度（shallow：当dataset复杂度较小时使用相对小的模型结构、regularization正则化：不知道dataset复杂度时优先使用大的网络结构）
* dropout算法
* 数据增强
* 提前终止训练

### 1. regularization
**L1-regularization**
$$ J_{(\theta)} = -\frac{1}{m} \sum\limits_{i=1}^m [y_iln\hat{y_i}+(1-y_i)ln(1-\hat{y_i})] + \lambda\sum\limits_{i=1}^n|\theta_i| $$

**L2-regularization 用的多**
$$ J_{(\theta)} = -\frac{1}{m} \sum\limits_{i=1}^m [y_iln\hat{y_i}+(1-y_i)ln(1-\hat{y_i})] + \frac{1}{2}\lambda\sum\limits_{i=1}^n||\theta_i||^2 $$

如果正则化的λ参数设置合适，可以在overfitting时不改变trainset的loss同时减小valset的loss
![regularization效果图](image/003026regularization.png)

>\> device = troch.device('cuda:0')
>\> net = MLP().to(device)
>\> optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)    &emsp;&emsp;#SGD()是求偏导的函数，weight_decay=0.01该参数用来设置L2-regularization的参数$\lambda=0.01$
>\> criteon = nn.CrossEntropyLoss().to(device)

### 2. 动量与学习率衰减
**Momentum：动量/惯性**
当前的参数下降方向为当前梯度下降+之前的梯度法方向，通过在优化器SGD()内设置参数即可使用momentum方法，或者adam()优化器内置默认使用该方法进行优化。

optimizer = torch.optim.SGD(model.parameters(), args\.lr, momentum=args.momentum, weight_decay=args.weight_decay)

**learning rate decay：学习率衰减**

**1.** 当loss维持了较长时间几乎不减小时，减小lr。

使用ReduceLROnPlateau()函数管理梯度信息以实现这种衰减方法。

optimizer = torch.optim.SGD(model.parameters(), args\.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min')
for epoch in xrange(args.start_epoch, args.epochs):
&emsp;train(train_loader, model, criterion, optimizer, epoch)
&emsp;result_avg, loss_val = validate(val_loader, model, criterion, epoch)
&emsp;scheduler.step(loss_val)      &emsp;&emsp;#loss没有下降的步数

**2.** 每过一定数量的epoch后，lr减少一些

scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)        &emsp;&emsp;#每过1000步，lr=lr*0.1
for epoch in range(100):
&emsp;scheduler.step()
&emsp;train(...)
&emsp;validate(...)

### 3. early stopping
![EarlyStopping](image/003027EarlyStopping.png)

### 4. dropout
![Dropout](image/003028dropout.png)
如上图，各神经元连接中每条线均有值为P的概率会断开。
torch中在两层网络间插入torch.nn.Dropout(概率值)函数实现dropout功能

**注：** torch.nn.Dropout()中P为断开的概率，tf.nn.Dropout()中P为保持连接的概率，在testset数据集中

net_droppout = torch.nn.Sequential(
&emsp;    torch.nn.Linear(784, 200),
&emsp;    torch.nn.Dropout(0.5),
&emsp;    torch.nn.ReLU(),
&emsp;    torch.nn.Linear(200, 200),
&emsp;    torch.nn.Dropout(0.5),
&emsp;    torch.nn.ReLU(),
&emsp;    torch.nn.Linear(200, 10),
)

### ==5. Stochastic Gradient Descent（随机梯度下降）==
==Stochastic的随机并非真正的随机分布，而是x~N(0, x)的正态随机分布==
==实际使用中是什么样呢？Gradient Descent是在所有的trainset上求loss并梯度下降，Stochastic Gradient Descent是在trainset上随机选出一部分数据求loss并对这个loss梯度下降==

---

<div STYLE="page-break-after: always;"></div>


# 九、CNN卷积神经网络
图片灰度和RGB值为0-255，一般会提前将照片转换为位图
Weight sharing 权值共享：同一层的每个卷积元的权重值（滤波）相同，CNN不是全连接，本层的神经元只和他的滤波器对应的神经元相连接
![Kernel操作](image/003029pt卷积操作.png)
Kernel的表达式记为 **$k_{(x, y)}$，其中x表示卷积核在x方向的偏移量，y表示卷积核在y方向的偏移量**

### 1. 常见卷积核
![常见卷积核](image/003030pt常见卷积核.png)
CNN卷积核运算公式为： $ F_{(x,y)} = \int I_{(x',y')}k{(x-x',y-y')}dxdy = \sum I*k $ ，其中F为卷积后输出结果，I为被卷积图像，k为卷积核。一个Kernel生成一个图层，例如一张[1,28,28]的图片经过三个不同Kernel后将生成[3,28,28]的图片。

$$\begin{cases}
Input\_channels: & 输入图片的RGB通道数\\
Kernel\_channels: & Kernel通道数即有几个Kernel,记为weight,kernet,filter\\
Kernel\_size: & Kernel大小例如3x3\\
Stride: & Kernel每次移动几格\\
Padding: & 图片外增加的空白px的行列数
\end{cases}$$

![Kernel示意图](image/003031pt中Kernel示意图.png)
**工作流程：**
photo --(kernel)--> | low-level feature | --> | mid-level kernel | --> | high-level feature | --> | trainable claeeifier | --> output

**Code：**
>layer = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)    &emsp;&emsp;#入口：（photo_channel，Kernel\_channels，Kernel\_size，Stride，Padding）
>x = torch.rand(1,1,28,28)
>out = layer.forward(x)
>out = layer(x)     &emsp;&emsp;#另一种求输出方法（调用torch实例），本质是调用__call__。建议用这个！

### 2. Pooling、upsample、ReLU
一个神经元配备函数有：Conv2d(卷积) --> batch normalization(批量归一化) --> Pooling --> ReLU

**Pooling：降维操作**
![MaxPooling](image/003032pt中MaxPooling操作.png)
>layer = nn.MaxPool2d(2, stride=2)      &emsp;&emsp;#入口（kernel_size，步长）
>output = layer(dataset)
>
>layer = F.avg_pool2d(x, 2, stride=2)   &emsp;&emsp;#入口（dataset，kernel_size，步长）

**upsample：增加px操作**
![upsample操作](image/003033pt中upsample操作.png)
>out = F.interpolate(x, scale_factor=2, mode='nearest')     &emsp;&emsp;#入口（photo's_tensor，放大倍数，采样模式）

**ReLU：**
![ReLU操作](image/003034pt中图像ReLU操作.png)
将图像中负数变成0，即将灰度图片中死黑亮度提高。

>layer = nn.ReLU(inplace=True)  &emsp;&emsp;#设为ture可节约内存
>out = layer(x)
>或者使用下面操作：
>out = F.relu(x)

### 3. batch normalization(批量归一化)
将神经网络的输入值限定在一定范围内，防止出现梯度离散现象。

>normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     &emsp;&emsp;#入口（统计得到的均值[R通道, G通道, B通道], 方差[R, G, B]），函数将会对数据进行标准正态

![求方差方法化简](image/003047求方差.png)

均值和方差是怎么统计的？BatchNorm是计算一个batch中所有照片同一个channel的均值和方差，其他norm计算方法如下：
$$ z^{(i)} = \frac{(x^{(i)}-μ)}{\sigma} $$

![不同norm的计算方法](image/003035pt图像的norm计算方法.png)

下图为Batch Norm计算过程，其中 $(\gamma, \beta)$ 需要根据梯度更新，$(\mu, \sigma)$ 为统计数据由计算得出，不需要更新。
![batchnorm方法](image/003036pt中batchnorm方法.png)

>x = torch.rand(100, 16, 784)   &emsp;&emsp;#trainset：100张照片，每张照片16通道、784px
>layer = nn.BatchNorma1d(16)    &emsp;&emsp;#照片有16个通道
>out = layer(x)
>layer.running_mean     &emsp;&emsp;#存放结果，x的每个channel的平均数，shape为[16]
>layer.running_var      &emsp;&emsp;#存放结果，为x的每个channel的方差，shape为[16]
 
![batchnorm代码](image/003037pt中batchnorm代码.png)
**在跑testset时，batch norm的 $ \mu 和 \sigma $ 用trainset的统计结果不用testset的 $ \mu 和 \sigma $**

### 4. 1x1 Convolution
1x1 size 的卷积核作用是对图像进行降维，减少图像的channel

### 5. LeNet_5、VGG、GoogLeNet、ResNet、Inception、DenseNet
**LeNet_5：** 数字识别用的该网络
![LeNet-5](image/003046LeNet5.jpg)
输入 ——> 卷积层 ——> 激活函数 ——> pooling ——> 卷积层 ——> 激活函数 ——> pooling ..... ——> 输出

**VGG： 运算量很大但是准确率不如新模型。**
计算量大，性能不如Inception和ResNet

**GoogLeNet：** 在神经元连接之间加入了几种不同的Kernel，识别图像的不同特征
![GoogLeNet](image/003038googlenet模型.png)

**ResNet：** 深度残差网络  
使用传统形式的神经网络结构时当网络层数增加，梯度会因为链式求导的误差积累而梯度离散，因此表现效果不好。ResNet在几个单元之间加入前向环节，当更深层次网络结构表现不好时可以使用前向环节将一部分神经元短路
![ResNet](image/003039ResNet.png)
**<center>ResNet基本单元</center>**

**上图网络的输出为：$H_{(x)} = F_{(x)} + x$，由于支路直接将输入输出短接，相当于引入正向前馈支路，因此神经网络实际训练的部分为：$F_{(x)} = H_{(x)} - x$，故称为残差网络。**

网络输出为 $ F_{(x)} + x = H_{(x)};--→ F_{(x)} = H_{(x)} - x $ ，因为网络 $F_{(x)} = H_{(x)} - x $ 有作差运算，因此称为残差网络。
ResNet的输入输出的channel不一致时，x使用size为1的卷积核使两个输出channel一致。

**DenseNet：** 
ResNet变种，增加了短接线，后面网络模块可以和前面所有的模块接触，将信息综合起来，但是如果后面的layer综合的不够好会导致网络的channel越来越大。
![DenseNet](image/003040DenseNet.png)

---

<div STYLE="page-break-after: always;"></div>


# 十、nn.Module模块
自己的函数继承nn.Module后可以使用torch的很多功能，也可以嵌套使用nn.Module。模块包含了很多网络层，调用nn.Module可以搭建网络，通过调用.call函数调用.forward()函数可以使用函数封装好的功能。

### 1. self\.net(x) = nn.Sequential()
nn.Sequential()：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。

nn.Sequential()函数将按照括号( ...... )中的函数顺序，从上到下执行，其中上一个函数的输出将作为下一个函数的输入， **必须确保上一层的输出和本层输入的tensor相同**

### 2. net.parameters()
.parameters()将会返回神经网络每一层的迭代结果。

>net = nn.Sequential(nn.Linear(4,2), nn.Linear(2,2))
>para = list(net.parameters())
>list(net.named_parameters())
>
>
>第一行net继承了nn.Module参数，第二行直接调用.parameters()便可以自动求解net参数，para.shape为[2, 4]，表示[网络输出, 网络层数]
>调用.named_parameters()将返回网络的所有参数，并自动给每个参数命名，以dict方式存储

### 3. save and load
目的：防止training过程中断电或系统需要重新training。

>net = Net()
>
>torch.save(net.state_dict(), 'ckpt.mdl')
>
>net.load_state_dict(torch.load('ckpt.mdl'))

### 4. train / test 状态切换
若神经网络中含有dropout或batch norm等，则在train与test时代码是不一样的，此时使用 train / test 状态切换

>net = Net()
>
>net.train()
>...
>
>net.eval()
>...
>

### 5. 自己定义类

**打平矩阵操作**
class Flatten(nn.Module):
&emsp;&emsp;def \_\_init\_\_(self):
&emsp;&emsp;&emsp;&emsp;super(Flatten, self).\_\_init\_\_()
&emsp;&emsp;def forward(self, input):
&emsp;&emsp;&emsp;&emsp;return.view(**input**.size(0), -1)
入口参数的input为自己定义的函数类别

**实际使用中的示例**
class TestNet(nn.Module):
&emsp;&emsp;def \_\_init\_\_(self):
&emsp;&emsp;&emsp;&emsp;super(TestNet, self).\_\_init\_\_()
&emsp;&emsp;&emsp;&emsp;self.net = nn.Sequential(nn.Conv2d(1, 16, strude=1, padding=1), nn.Maxpool2d(2, 2), Flatten(), nn.Linear(1\*14\*14, 10))
&emsp;&emsp;def forward(self, x):
&emsp;&emsp;&emsp;&emsp;return self.net(x)

### 6. 自定义网络层

使用 nn.Parameter(tensor) ，将会把矩阵tensor添加到nn.parameters()容器中，在求解过程中可以自动被优化器优化

---

<div STYLE="page-break-after: always;"></div>


# 十一、数据增强(增加数据多样性)
**导入 torchvision 包** ，其中的transform.Compose()用途和nn.parameters()一样，作为一个容器将图像处理程序放进去

**在小数据下：**  
* 减少模型参数数量small network capacity
* 对数据正则化regularization
* 数据变换data augmentation

### 1. data augmentation
**Recap**
![Recap](image/003041Recap.png)

**Flip：翻转**
>.RandomHorizontalFlip()&emsp;&emsp;#随机水平翻转
>.RandomVerticalFlip()&emsp;&emsp;#随机竖直反转
>.ToTensor()    &emsp;&emsp;#将Flip后的数据转移到totensor上
>
>![Flip](image/003042Flip.png)


**Rotate：旋转**
>.RandomRotation(15)&emsp;&emsp;#在(-15,15)度范围内随机旋转
>.RandomRotation([90, 180, 270])&emsp;&emsp;#在90,180,270度中随机选一个度数旋转
>.ToTensor()
>
>![Rotate](image/003043Rotate.png)

**Scale：缩放** (以图片中心为中心，根据输入值的大小，放大这个大小的画面)
>.Resize([32, 32])  &emsp;&emsp;#将中心画面的横竖方向的 32%,32% 大小的图像放大
>.ToTensor()    &emsp;&emsp;#将Flip后的数据转移到totensor上
>
>![Scale](image/003044Scale.png)

**Crop Part：部分裁剪** 随机裁剪一部分后对裁剪部分补0
>.RandomCrop([28, 28])  &emsp;&emsp;#随机删去图像中28*28px大小区域
>.ToTensor()
>
>![Crop Part](image/003045Crop.png)

**Noise：增加噪声**
torch中没有包提供该功能

**GAN** 新技术！

---
<div STYLE="page-break-after: always;"></div>


# 十二、数据集
### 1. MNIST dataset
内容为0-9的手写数字

### 2. CIFAR-10 dataset
两个版本CIFAR-10 、 CIFAR-100，是一个图像物体分类数据集
CIFAR-10包含10类物体，每个类别有6000张照片


---
<div STYLE="page-break-after: always;"></div>



# 十三、LeNet-5、ResNet Project
### 1. LeNet-5
网络结构图：
![LeNet-5](image/003046LeNet5.jpg)

Convolutions：卷积层。  Subsampling：池化层

LeNet-5介绍：https://zhuanlan.zhihu.com/p/635275872

### 2. ResNet
网络结构图：
![ResNet](image/003039ResNet.png)

对数据集采取transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 这个 BatchNormalize 后loss明显单调下降，acc明显单调上升！！！
