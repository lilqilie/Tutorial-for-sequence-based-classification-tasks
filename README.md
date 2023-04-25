
# 一   NCBI Downloader

目录：`download_seq_from_ncbi` 

该项目为NCBI数据下载程序，用于从NCBI数据库下载指定的生物信息数据。本程序需要一个配置文件和一个CSV文件来指定需要下载的文件列表和下载路径。

## 安装依赖

程序依赖环境：

- pandas
- wget
- Python 3.x

你可以通过以下命令安装：

```
pip install pandas
pip install wget
```

## 下载设置

下载设置可以在download_config.py文件中进行修改。

- `Config.input_path`：输入CSV文件路径。
- `Config.output_path`：输出文件路径。
- `Config.fail_log_path`：下载失败日志输出路径。
- `Config.csv_col`：CSV文件中指定需要下载的文件列的名称。
- `Config.file_type`：下载文件的后缀名，需要手动根据所需数据类型定义。
- `Config.retry_times`：下载失败后的重试次数。
- `Config.cpu_worker_num`：CPU工作线程数。

## 输入文件格式

程序需要一个CSV文件作为输入，用于指定需要下载的文件列表和下载路径。CSV文件应包含以下列：

- 文件名
- 下载地址

文件夹中的NCBI_22June_RefSeq_32927_Complete_1NP_2P_taxnonmy.csv为基因组信息文件，详情如下：

![Untitled](readme%20da37e461a6eb43c28a7d2e6c05c9b0de/Untitled.png)

其中包括ftp_path，为ftp下载路径。

## 输出文件格式

下载成功的文件将存储在指定的输出路径中，下载失败的文件将被记录在指定的下载失败日志输出路径中。

## **使用方法**

1. 在 **`download_config.py`** 中设置下载相关参数，包括输入输出路径、线程数、下载文件后缀、CSV 文件中的下载链接列名、下载失败后的重试次数、下载失败日志输出路径等。
2. 在CSV 文件中需要包括下载的链接列表。
3. 在 **`NCBI_download.py`** 中运行主程序，开始下载。

在终端中进入NCBI_download.py所在目录，输入以下命令运行程序：

```
python NCBI_download.py
```

## Notes

- 该程序使用多线程下载，可以加速文件的下载。你可以通过修改Config.cpu_worker_num来控制CPU工作线程数。
- 下载的数据将保存在指定的输出路径中，请确保该路径可用并具有足够的存储空间。
- 下载过程中可能会发生连接错误或下载失败等情况，程序会自动重试。如多次尝试后仍无法下载，程序会将下载失败的信息保存在指定的日志文件中。

# 二   计算kmer频率

目录：`cal_kmer_freqs`

## **k-mer的定义**

k-mer是指长度为k的连续子串（或片段）序列，其中k是一个正整数。在DNA序列中，k-mer由四种核苷酸（腺嘌呤A、胸腺嘧啶T、鸟嘌呤G和胞嘧啶C）构成；在蛋白质序列中，k-mer由20种氨基酸构成。例如，当k=3时，在DNA序列“ATCGCGATCG”中，所有长度为3的k-mer为：“ATC”，“TCG”，“CGC”，“GCG”，“CGA”和“GAT”。

## **k-mer频率的定义**

k-mer频率的计算是“每个k-mer出现的次数”除以“所有可能的k-mer数”。具体而言，对于长度为n的序列，总的可能的k-mer数是n-k+1，其中k是k-mer的长度。因此，k-mer频率的计算公式是：

```
k-mer频率 = k-mer出现的次数 / (n - k + 1)
```

例如，在长度为10的DNA序列“ATCGCGATCG”中，k=3时，“ATC”的频率为2/8=0.25，“TCG”的频率为2/8=0.25，“CGA”的频率为1/8=0.125，如下所示：

| k-mer | 频率 |
| --- | --- |
| ATC | 0.25 |
| TCG | 0.25 |
| CGA | 0.125 |
| GCG | 0.125 |
| CGC | 0.125 |
| GAT | 0.125 |

对于大规模的序列数据，计算k-mer频率可以帮助研究人员了解序列的特征，比如确定序列的重复性、组成和结构等信息。

## 代码介绍

包含了合并fasta文件并计算kmer频率的功能。从指定的fasta目录下读取多个fasta文件，并将它们合并成一个文件。然后，对合并后的序列计算kmer频率，并将结果保存到指定的文件中。

## **输入文件格式**

输入文件为标准的FASTA格式，例如：

```bash
>Seq_1
ATCGATCGATCG
>Seq_2
CGATCGATCGAT
```

## **输出文件格式**

输出文件为CSV格式，文件名为**`<output_file_prefix>_kmer_freqs.csv`**，例如：

```
AAA,AAC
0.0135470497,0.0129958199
0.0025780647,0.0029258807
...
```

每一列是每个k-mer在每个序列中的频率。

## 使用方法

1. 修改`config_set()`函数中的配置项：
    - `fasta_path`：指定要合并的fasta文件所在目录的路径。
    - `combined_file`：指定合并后的文件保存的路径和文件名。
    - `num_procs`：指定用于计算kmer频率的CPU核心数。
    - `ks`：指定要计算的kmer频率的大小。
    - `freqs_file`：指定kmer频率计算结果保存的文件路径和文件名。
2. 运行`cal.py`文件，即可自动执行合并fasta和计算kmer频率的操作。

## 注意事项

1. 确保Python 3.x已经正确安装并配置。
2. 确保所需的第三方Python库已经安装。这些库包括：
    - `pandas`
    - `numpy`
    - `biopython`
    - `itertools`
    - `multiprocessing`
    - `tqdm`
3. 如果需要使用其他kmer大小，请在`config_set()`函数中添加或修改`ks`列表。例如，如果要计算9-mer频率，可以将`ks`列表修改为：
    
    ```
    'ks': [3, 4, 5, 6, 7, 8, 9],
    
    ```
    
4. 如果需要对其他类型的fasta文件进行处理，请修改`combine_fna.py`文件中的`get_seq_id()`函数和`get_seq()`函数。这些函数分别用于获取序列的名称和序列内容。默认情况下，这些函数假定fasta文件中每条序列都有一个以`>`开头的名称行和一个包含序列的文本行。
5. 在计算kmer频率时，程序使用多进程并行计算，以提高计算效率。可以通过修改`config_set()`函数中的`num_procs`参数来控制使用的CPU核心数。建议根据计算机配置和任务复杂度来合理调整该参数。

# 三   DeepResCross 组合网络

目录：`DeepResCrossNet`

这段代码是一个深度交叉网络的PyTorch实现，包含了`ResidualBlock、Deep、Cross`和`DCiPatho`这些类的定义。

## ResidualBlock

ResidualBlock是残差块的定义，包含了两个线性层和一个ReLU激活函数，forward方法中将输入通过线性层和ReLU激活函数后与原输入相加并再次经过ReLU激活函数。

## Deep

Deep是深度网络的定义，包含多个线性层和BatchNorm层，在初始化方法中将输入维度和深度网络的每一层连接起来构建出一个Sequential网络，forward方法中将输入通过整个网络并返回输出。

## Cross

Cross是交叉层的定义，包含多个交叉层，每个交叉层都由一个线性层、一个BatchNorm层和一个ReLU激活函数组成。在初始化方法中，将输入维度和交叉层的个数作为参数，然后初始化交叉层的权重和偏置，并将它们作为类的参数保存。forward方法中将输入进行交叉，并在每一层后都添加一个BatchNorm层。

## DCiPatho

DCiPatho是整个模型的定义，包含了多个ResidualBlock、一个Deep、一个Cross和几个线性层。在初始化方法中，定义了模型的各层，并在forward方法中按照模型的结构将输入进行处理并输出。

## 网络的不同组合

在`network.py`文件中，通过对类DCiPatho中forward函数的注释，可以实现不同网络的组合。例如注释掉以下代码，即可消除掉deepNet的网络，仅保留resNet和CrossNet。

需要注意的是，更改后必须修改对应的输出层的维度和特征融合层的操作。

```bash
# deep_out = self._deepNet(x)
# deep_out = torch.relu(self.out_layer2(deep_out))

## 对应修改
self._final_linear = nn.Linear(config.end_dims[-1], 1)
final_input = torch.cat([res_out, cross_out], dim=1)
```



## 模型及训练参数设置

在config.py中进行设置。各参数说明如下：

- patho_path: 路径名，指向一个npy文件，包含了病原体的数据。
- nonpatho_path: 路径名，指向一个npy文件，包含了非病原体的数据。
- test_path: 字符串类型，表示测试数据的路径。
- hidden_layers: 列表类型，表示ResNet模块的隐藏层大小。
- deep_layers: 列表类型，表示DeepNet模块的隐藏层大小。
- num_cross_layers: 整数类型，表示CrossNet层数。
- end_dims: 列表类型，表示网络的最终维度。
- out_layer_dims: 整数类型，表示输出层的大小。
- val_size: 浮点数类型，表示验证集的比例。
- test_size: 浮点数类型，表示测试集的比例。
- random_state: 整数类型，表示随机数种子。
- num_epoch: 整数类型，表示训练的轮数。
- patience: 整数类型，表示EarlyStopping的耐心值。
- batch_size: 整数类型，表示训练的批次大小。
- Dropout: 浮点数类型，表示Dropout的概率。
- lr: 浮点数类型，表示学习率。
- l2_regularization: 浮点数类型，表示L2正则化的系数。
- device_id: 整数类型，表示GPU设备的ID。
- use_cuda: 布尔类型，表示是否使用GPU。
- save_model: 布尔类型，表示是否保存模型。
- output_base_path: 字符串类型，表示模型输出的路径。
- best_model_name: 字符串类型，表示最佳模型的文件名。

## 模型训练与验证

该程序是一个基于PyTorch框架的致病菌分类器，旨在根据序列的Kmer频率特征进行分类为致病菌与非致病菌。

## 运行环境

该程序运行环境为Python 3.7及以上版本，并依赖以下库：

- PyTorch 1.9.0
- pandas
- sklearn

## 使用说明

### 训练模型

运行`main.py`函数，即可进行模型训练。其中，该函数使用了5折交叉验证的方式进行模型训练，并输出每折交叉验证的模型性能指标（包括ROC，ACC，F1，MCC）。

### 测试模型

运行`test_best_model()`函数，即可对模型进行测试，并输出模型的性能指标。

### 保存模型

模型训练完成后，会自动将表现最好的模型保存到本地。

# 四   GNN毒性**分类任务**

目录：`GNN_demo` 

## **概述**

这是一个使用 PyTorch Geometric 库实现的简单的 GNN 网络，用于将化学 SMILES 分子图进行分类。该网络是基于图神经网络 (GNN) 技术的一种应用。GNN 是一种适用于图数据的深度学习方法，已经在许多领域得到了广泛应用，如推荐系统、社交网络、语义分割等。

SMILES 分子图是一种常用的化学分子表示方法，其中每个原子和化学键都表示为图的节点和边。使用 GNN 对 SMILES 分子图进行分类任务的目的是为了将化学分子图区分为不同的类别。在此过程中，GNN 可以通过学习分子中的节点和边之间的关系来提取分子的特征，并对分子进行分类。

## 数据集

该模型使用了Tox21数据集，该数据集由US Environmental Protection Agency提供，其中包含了数千个分子的毒性数据。分子是通过化学结构的SMILES表示进行编码的。在数据集预处理中，使用RDKit将SMILES编码转换为分子的特征向量表示。在代码中将自动下载。

## **代码实现**

在这个 GNN 网络中，使用了两个 GCNConv 层和一个全连接层进行前向传递。GCNConv 层是一种用于图卷积的模块，可以有效地处理节点和边之间的关系。全连接层用于将图形级别的特征映射到输出标签空间。ReLU 激活函数和全局最大池化操作被用于提取图级别的特征，log_softmax 激活函数用于分类。

SMILES 分子图可以通过 **`smiles_to_graph`** 函数将其转换为图形表示，该函数接受 SMILES 字符串作为输入，并返回 PyTorch Geometric 的 **`Data`** 对象。在将 SMILES 分子图传递给 GNN 进行分类之前，需要将其转换为适当的格式。在这个Demo中，使用了 Morgan 指纹作为节点特征，使用邻接矩阵作为边索引。

## 用法

使用以下命令来训练和验证模型，训练过程将输出损失和准确率：

```
python main.py

```

