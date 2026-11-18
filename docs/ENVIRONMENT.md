# 环境配置说明

## 1. 环境概览

本说明依据仓库源码及已有结果文件编写。本次完成文档和依赖静态核查，安装、设备检查和训练命令供运行者执行，实际环境与运行结果需另行记录。

| 项目 | PyTorch 实现 | MindSpore 实现 |
| --- | --- | --- |
| 工作目录 | `pytorch_version/` | `ICT_MindAligner_MindSporeVersion/` |
| 训练入口 | `main.py` | `main_mindspore.py` |
| 设备硬编码 | `torch.device('cuda', 0)` | `ms.set_device("Ascend", 0)` |
| 数据管线 | PyG `InMemoryDataset` | MindSpore `NumpySlicesDataset` |
| 预处理缓存 | `./processed/` 下的 `.dataset` | `./processed_ms/` 下的 `.npz` |
| 输出目录 | `./result/` | `./result/` |
| 当前限制 | 数据路径与 CUDA 设备需手动适配 | 数据路径与 Ascend 环境需适配，MMD 文件为空 |

分别使用独立虚拟环境。两份 `requirements.txt` 根据对应版本全部 Python 文件的直接第三方 import 编写，标准库和仓库本地模块由 Python 或仓库提供。PyTorch 清单包含实际导入的 Matplotlib，MindSpore 源码未导入 Matplotlib。`torch-geometric` 的导入名为 `torch_geometric`，`scikit-learn` 的导入名为 `sklearn`。

仓库缺少训练环境锁定记录，依赖清单使用未锁定版本。实际训练环境应提供 Python、框架及全部依赖版本、操作系统与架构、GPU/NPU 型号、驱动、CUDA 或 CANN 信息，并关联实验批次、代码提交和数据版本。

### 实际训练环境记录

下表供维护者根据实际训练机器和环境检查结果填写。源码中的设备目标及路径仅表示当前配置，实际训练环境信息仍需核验。

| 记录项 | PyTorch 训练环境 | MindSpore 训练环境 |
| --- | --- | --- |
| 操作系统及架构 | 待确认 | 待确认 |
| Python 版本 | 待确认 | 待确认 |
| 框架版本 | PyTorch、PyG 版本待确认 | MindSpore 版本待确认 |
| CUDA/CANN 版本 | CUDA 版本待确认 | CANN 版本待确认 |
| GPU/NPU 型号 | GPU 型号待确认 | NPU 型号待确认 |
| 驱动及固件版本 | 待确认 | 待确认 |
| 环境验证状态 | 待验证 | 待验证，当前训练入口受空 MMD 文件阻塞 |

第 11 节提供环境检查命令。完成检查后记录执行日期、代码提交与实际结果，再更新验证状态。

## 2. PyTorch 环境安装

先选择同时满足 PyTorch、PyG 和科学计算依赖要求的 Python 解释器。在仓库根目录创建并激活独立环境，下列环境目录为后续安装时生成。

```bash
python -m venv .venv-pytorch
```

Windows PowerShell 激活命令如下。

```powershell
.\.venv-pytorch\Scripts\Activate.ps1
```

Linux Bash 激活命令如下。

```bash
source .venv-pytorch/bin/activate
```

根据操作系统、Python 和可用 NVIDIA 驱动，在官方安装选择器中选择 CUDA 构建并执行生成的安装命令，然后安装项目依赖。仓库直接导入 `torch`，未直接使用 `torchvision` 或 `torchaudio`，依赖清单按实际 import 列出。[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)

```bash
cd pytorch_version
# 在此先执行官方选择器生成的 PyTorch 安装命令
python -m pip install -r requirements.txt
python -m pip check
```

清单含 `torch` 和 `torch-geometric`。如已按指定版本安装，需核对 pip 的解析结果与选定组合一致。未锁定清单用于列出依赖，兼容版本需由实际环境验证后记录。

## 3. PyTorch Geometric 依赖兼容要求

先安装并检查 PyTorch，再安装 PyG。检查项目使用的 `torch_geometric.data.DataLoader`、`InMemoryDataset`、`DenseGCNConv`、`to_dense_batch` 等接口在所选版本中可用。源码还使用 `torch.load(..., weights_only=False)`，选择版本时需确认该参数受支持。

PyG 基础安装可通过 `torch-geometric` 获取。扩展包如 `pyg-lib`、`torch-scatter`、`torch-sparse` 应按实际算子需求安装，其 wheel 需匹配 PyTorch、`torch.version.cuda`、Python ABI、操作系统及架构。根据官方兼容表选择扩展包及下载地址。[PyG 官方安装与兼容说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

当前源码未直接 import 上述扩展包，因此本次清单列出直接依赖。出现算子缺失或二进制加载错误时，先检查 PyG 文档中对应版本的依赖要求。

## 4. MindSpore Ascend 环境安装

在目标 Ascend 机器上，根据所选 MindSpore 发行版核对支持的操作系统、CPU 架构、Python、Ascend 设备、驱动、固件和 CANN 组合，按官方指南完成底层环境安装，再执行对应的 MindSpore 安装命令。具体组合应由实际训练环境提供。[MindSpore 官方安装入口](https://www.mindspore.cn/install)

在仓库根目录、满足该发行版要求的 Python 环境中创建独立环境。

```bash
python -m venv .venv-mindspore
source .venv-mindspore/bin/activate
cd ICT_MindAligner_MindSporeVersion
# 在此先执行适配目标 Ascend 平台的 MindSpore 官方安装命令
python -m pip install -r requirements.txt
python -m pip check
```

以上为目标 Linux Ascend 环境的 Bash 命令。云平台预装镜像可使用其匹配的框架环境，执行前应核对当前解释器与软件栈版本。`requirements.txt` 中的 `mindspore` 仅声明直接依赖，具体发行包、索引或 wheel 由平台官方安装说明确定。需确认所选版本提供源码使用的 `ms.set_device`、`ms.value_and_grad`、`nn.Dropout(p=...)` 等 API。

依赖安装完成后，训练入口仍受本地 MMD 实现缺失限制，详见第 9 节。

## 5. Python 版本与操作系统要求

原 README 写有 Python 3.8 或以上，该信息可作为历史说明。当前仓库缺少精确兼容组合，部署时应以选定 PyTorch/PyG 或 MindSpore 发行版的 Python 支持范围交集为准。

PyTorch 版本需在所选发行版支持的 Windows 或 Linux 等平台中部署，并配置源码指定的 CUDA 设备。MindSpore 当前入口面向 Ascend，部署到其支持的 Linux/架构组合。具体 Python 小版本、操作系统版本和硬件型号均待实际环境确认。CPU 环境可用于部分静态检查，完整训练仍需处理源码中的设备设置及算子支持。

## 6. CUDA 与 CANN 相关配置说明

### CUDA

使用 `nvidia-smi` 检查设备与驱动，使用 `torch.version.cuda` 确认 PyTorch 的 CUDA 构建版本，再核对 PyG 扩展包。系统 `nvcc --version` 反映本地 Toolkit，安装预编译包和编译扩展时应分别检查各自要求。`CUDA_VISIBLE_DEVICES` 会影响进程看到的逻辑设备编号，源码索引 0 指进程内的第一张可见 GPU。[PyTorch 安装说明](https://pytorch.org/get-started/locally/)

### CANN

在目标机器按对应版本指南加载 CANN 环境脚本，脚本位置以实际安装为准。下列路径为占位示例，需替换后执行。

```bash
source /path/to/ascend-toolkit/set_env.sh
npu-smi info
```

检查 `PATH`、`LD_LIBRARY_PATH` 和 `ASCEND_HOME_PATH` 等变量是否满足所选版本要求。CANN、Ascend 驱动与固件的安装独立于 Python requirements，加载环境后再检查 MindSpore 的 Ascend 可用性。[MindSpore 安装与平台配置](https://www.mindspore.cn/install)

## 7. SEED 数据目录和 label.mat 要求

仓库未提供 SEED 数据，需按数据集使用要求自行申请和准备。以下展示外部数据的预期布局，尖括号内容为文件名占位。

```text
ExtractedFeatures/
├── label.mat
└── <被试与 session 对应的特征文件>.mat
```

- 默认读取 15 个被试，每被试 3 个 session，共 45 个特征 `.mat` 文件，每文件处理 15 个试次。
- `label.mat` 必须含 `label` 键，对应每个 session 的 15 个试次标签，代码将其复制三次并转为三类 one-hot 标签。
- 每个特征文件需有 15 个名称含 `de_movingAve` 的字段。代码按字段顺序遍历，每个字段经 `transpose(0, 2, 1)` 后应为 `(62, 5, T)`，要求 `T <= 265`，随后填充到 `(62, 5, 265)` 并展平到 `(62, 1325)`。
- 两套代码对目录排序后按每三个文件分组，需人工确认文件顺序确实对应同一被试的三个 session，字段顺序与标签一致。
- PyTorch 使用 `glob('*')`，依赖前 45 个排序条目为特征文件，未显式过滤 `label.mat` 和其他条目。MindSpore 仅过滤路径中包含 `label.mat`、`readme.md` 的条目。准备目录时应检查实际排序，避免其他文件或子目录进入特征读取范围。

当前加载器针对 SEED 的三分类特征结构，原 README 提及的 SEED-IV 及意识状态数据需另行确认适配方案。

## 8. .env.example 变量解释

**当前源码没有 `.env` 读取逻辑，模板仅作配置参考。复制为 `.env` 或设置同名变量后，当前路径与训练参数仍需由运行者手动配置。** 模板中的设备字符串用于表达配置意图，需转换为源码对应 API 参数。

| 变量 | 模板示例 | 手动配置含义 |
| --- | --- | --- |
| `SEED_FEATURES_DIR` | `/path/to/SEED/ExtractedFeatures/` | 两套数据管线 `get_data()` 的 `path` |
| `PYTORCH_DEVICE` | `cuda:0` | 两处 `torch.device('cuda', 0)` 的设备类型与索引 |
| `MINDSPORE_DEVICE` | `Ascend:0` | `ms.set_device("Ascend", 0)` 的目标与索引 |
| `NUM_SUBJECTS` | `15` | 对应版本主入口与数据管线的 `subjects` |
| `EPOCHS` | `200` | 对应主入口的 `epochs` |
| `RANDOM_SEED` | `42` | 对应主入口 `set_random_seed(42)` 的调用参数 |

15、200、42 均来自当前源码默认值。实际路径需使用本机数据位置，模板包含通用占位路径。

## 9. 现有 Python 代码中需要手动修改的数据路径和设备位置

下表说明运行者后续部署时的配置位置，本次编辑范围为文档和依赖清单。

| 文件与位置 | 当前源码内容 | 配置方法 |
| --- | --- | --- |
| `pytorch_version/datapipe.py` 的 `get_data()` | `path = 'E:/EEG_data/SEED/ExtractedFeatures/'` | 改为本机 SEED 特征目录，保留末尾 `/`，因为标签路径使用 `path + 'label.mat'` 拼接 |
| `pytorch_version/main.py` 全局 `device` | `torch.device('cuda', 0)` | 按进程可见 GPU 设置索引，并核对随机种子函数中的 CUDA 调用 |
| `pytorch_version/Net.py` 全局 `device` | `torch.device('cuda', 0)` | 与主入口保持一致，当前全局变量在该文件中未被后续代码引用 |
| `ICT_MindAligner_MindSporeVersion/datapipe_mindspore.py` 的 `get_data()` | `path = '/home/ma-user/work/SEED_Dataset/ExtractedFeatures/'` | 改为目标机器特征目录，确认其中的 `label.mat` 可读 |
| `ICT_MindAligner_MindSporeVersion/main_mindspore.py` 运行环境设置 | `ms.set_context(mode=ms.PYNATIVE_MODE)`、`ms.set_device("Ascend", 0)` | 核对 Ascend 设备索引与所装 MindSpore API 支持 |

每套主入口和数据管线均定义 `subjects = 15`，调整被试数时需同步核对这两处及数据分组。两套主入口均设 `epochs = 200`，并调用 `set_random_seed(42)`。修改随机种子配置时应核对函数默认参数和实际调用值。

数据缓存依赖被试数、版本号及折号。PyTorch 构建函数按训练缓存存在与否跳过处理，MindSpore 仅检查第 0 折训练缓存就可能跳过整个构建阶段。重用缓存前应确认所有折及混合域文件完整、配置与数据来源一致。

### 实现差异与已知限制

- **MindSpore MMD 缺失**。`MMD_mindspore.py` 当前为 0 字节，未定义 `main_mindspore.py` 导入的 `mmd_rbf`，训练入口会在导入阶段受阻。安装依赖无法补齐该本地模块，需后续代码工作处理。
- **GRL 系数更新**。两套 `Net` 文件均定义 `Progressive_GRL.update_alpha`，系数初始化为 0，但当前训练入口未调用该更新方法。渐进混合域调度已在入口中使用，GRL 系数的渐进更新仍需另行核实和处理。
- **选边优化差异**。PyTorch 入口包含奖励和策略损失计算，MindSpore 入口显式从优化器中排除了名称含 `rl_agent` 的参数，且总损失未加入对应策略损失。两套实现的训练行为存在差异。
- **模型权重保存**。PyTorch 入口保存指标与曲线数据，`torch.save` 用于预处理缓存，当前没有模型权重保存逻辑。MindSpore 使用固定的 `best_model_cvXX.ckpt` 文件名保存权重，再次运行会写入同名路径，后续实验应保留已有结果。
- **历史结果对应关系**。仓库已有 MindSpore 日志和 checkpoint，其生成时的代码、环境及批次对应关系待确认，当前源码仍受空 MMD 文件限制。

## 10. 实际运行命令和工作目录

以下为源码对应的训练入口，均要求先完成环境、数据及手动配置。本次未执行训练。

### PyTorch

在仓库根目录、已激活 PyTorch 环境时执行。

```bash
cd pytorch_version
python main.py
```

`main()` 调用 `build_dataset(subjects)`，自动使用当前工作目录下的 `processed/`、`result/`。单独运行 `python datapipe.py` 只加载定义，无法触发数据构建。训练入口保存指标与绘图数据，源码中的 `torch.save` 保存预处理缓存。

### MindSpore

在独立仓库副本的根目录、已激活目标 Ascend 环境时执行。入口当前受空 `MMD_mindspore.py` 限制，须由后续代码工作补齐后才能继续验证。

```bash
cd ICT_MindAligner_MindSporeVersion
python main_mindspore.py
```

入口调用数据构建并使用当前工作目录下的 `processed_ms/`、`result/`。CSV 文件会选择递增编号，checkpoint 使用固定的 `best_model_cvXX.ckpt` 文件名，后续在独立副本中开展实验可保留已有结果。两套训练入口均在模块顶层包含结果目录或日志初始化操作，环境检查时应使用下节独立命令。

## 11. 环境检查命令

下列命令由运行者在目标环境执行，本次未执行依赖导入与设备测试。检查成功仅表明对应安装或算子检查完成，完整实验仍需独立验证。

### 通用检查

```bash
python --version
python -m pip --version
python -m pip check
python -m pip freeze
python -c "import platform; print(platform.platform()); print(platform.machine())"
python -c "import numpy, pandas, scipy, sklearn, tqdm; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, tqdm.__version__)"
```

### PyTorch 环境

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import torch_geometric, matplotlib; from torch_geometric.data import DataLoader, InMemoryDataset; from torch_geometric.nn import DenseGCNConv; from torch_geometric.utils import to_dense_batch; print(torch_geometric.__version__, matplotlib.__version__)"
```

### MindSpore Ascend 环境

```bash
npu-smi info
python -c "import mindspore as ms; print(ms.__version__); print(hasattr(ms, 'set_device'))"
python -c "import mindspore as ms; ms.set_context(mode=ms.PYNATIVE_MODE); ms.set_device('Ascend', 0); print((ms.Tensor([1.0], ms.float32) + ms.Tensor([2.0], ms.float32)).asnumpy())"
```

最后一条需在完成软件栈配置且所选版本支持 `set_device` 的 Ascend 设备上执行。

### SEED 数据检查

将下列路径替换为真实特征目录，在对应版本工作目录或其他只读检查目录执行。命令只读取数据，输出全部条目用于核对排序。

```bash
python -c "from pathlib import Path; import scipy.io as sio; p=Path('/path/to/SEED/ExtractedFeatures/'); d=sio.loadmat(str(p/'label.mat')); print(d['label'].shape); print(d['label']); print('\n'.join(str(f) for f in sorted(p.glob('*'))))"
```

## 12. 常见报错与排查

| 现象 | 排查位置与处理方向 |
| --- | --- |
| `No module named ...` | 确认当前虚拟环境、`python -m pip` 对应解释器及版本目录下的依赖清单 |
| `Torch not compiled with CUDA enabled` 或 CUDA 不可用 | 检查 PyTorch 安装构建、驱动、可见 GPU 和两处 CUDA 配置 |
| `invalid device ordinal` | 检查设备索引以及 `CUDA_VISIBLE_DEVICES` 映射 |
| PyG 出现 `undefined symbol`、DLL 加载失败或算子缺失 | 核对 PyTorch/PyG/扩展包的版本、CUDA 构建、Python ABI 和平台 |
| `torch.load` 不接受 `weights_only` | 核对 PyTorch 版本是否支持源码使用的参数 |
| MindSpore 缺少 `set_device` 等属性 | 根据所选发行版 API 文档核对安装版本与源码兼容性 |
| Ascend 初始化或动态库加载失败 | 检查 `npu-smi info`、驱动与固件、CANN 兼容组合和环境脚本 |
| `cannot import name 'mmd_rbf' from 'MMD_mindspore'` | 当前本地 MMD 文件为空，需后续补齐实现，依赖安装无法解决 |
| 找不到 SEED 数据或 `label.mat` | 核对数据管线中硬编码路径、文件权限和 PyTorch 路径末尾分隔符 |
| `KeyError: 'label'` 或试次字段索引越界 | 检查标签键、15 个 `de_movingAve` 字段及实际文件是否为所需 SEED 特征 |
| `loadmat` 读取非 MAT 文件失败 | 检查目录排序、前 45 个条目以及加载器的过滤范围 |
| 数据填充或 reshape 报错 | 核对 62 通道、5 频带、时间长度不超过 265、标签数量及文件分组 |
| 缓存缺失或仍使用旧数据 | 检查 `processed/` 或 `processed_ms/` 全部折文件及其配置来源，在独立副本中重新准备所需缓存 |
| 改了 `.env` 后配置未生效 | 按第 9 节手动配置路径、设备和参数，现有源码尚未读取模板变量 |

返回项目说明。[根目录 readme.md](../readme.md)
