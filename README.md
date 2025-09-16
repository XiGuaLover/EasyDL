## EasyDL: 易上手的深度学习模型训练仓库

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-Required-red" alt="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch%20Lightning-Required-orange" alt="PyTorch Lightning">
  </p>

EasyDL 是一个用于深度学习模型训练的工具，针对部分开源仓库代码进行优化，目的是降低广大同学上手难度和学习难度。
基于 Python、PyTorch 和 PyTorch Lightning 构建，旨在简化数据加载、模型训练、验证和测试的流程。项目目前内置了部分模型，如 ConvLSTM、PredRNN 系列，方便用户快速上手和实验。

## 🚀 快速开始

### 安装

1. 克隆项目仓库到本地：

    - `git clone` <repository-url>
    - `cd easyDL`

2. 安装项目依赖：
    - pip install -r requirements.txt （后续整）

### 运行

1. 配置模型: 在 _utils/ConfigData.py_ 中修改 _SuperParams.runNets_ 列表，选择要运行的模型。
    - 提示：如运行默认配置，请修改 _utils/configDatas/configDatasets.py_ 文件，指定数据集路径。
2. 启动训练:
    - 在**项目根目录**下运行 _run.py_ 文件开始训练：
        - ` python run.py`
    - 可替换：当前我常用的做法为：
        - `nohup python run.py > "./logs/test-$(date +%Y%m%d-%H).log" 2>&1 &`

## 📁 项目结构

    1  easyDL/
    2  ├── datasets/                     # 数据集处理模块
    3  │   ├── SequenceTyphoonDataset.py
    4  │   ├── TyphoonTimeSeriesDataset.py
    5  │   ├── TyphoonTimeSeriesFilterDataset.py
    6  │   └── TyphoonTimeSeriesModule.py
    7  ├── libs/
    8  │   └── pyphoon2/
    9  ├── logs/                         # 日志文件
    10 ├── models/                       # 模型定义
    11 │ ├── components/
    12 │ │ └── Component.py
    13 │ ├── cores/
    14 │ └── lightnings/
    15 │ └── LightningBaseSequence.py
    16 ├── results_test/
    17 ├── utils/
    18 ├── run.py                        # 主运行文件
    19 └── README.md

## 🤝 贡献

我们欢迎任何形式的贡献！如果您有任何建议、功能请求或发现任何问题，请提交 Issue 或 Pull Request。
