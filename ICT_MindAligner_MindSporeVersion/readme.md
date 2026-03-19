<<<<<<< HEAD
# 🌟 Progressive Multi-Domain Adaptation Network with Reinforced Self-Constructed Graphs 🚀

Welcome to the official repository for our paper **"A Progressive Multi-Domain Adaptation Network with Reinforced Self-Constructed Graphs for Cross-Subject EEG-Based Emotion and Consciousness Recognition"**, published in *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*, 2025! 🎉

This repository contains the complete implementation of our innovative framework for cross-subject EEG-based emotion and consciousness recognition. 🧠✨

## 📖 Overview

Our project introduces a cutting-edge **Progressive Multi-Domain Adaptation Network** that leverages reinforced self-constructed graphs to address domain shift and subject variability in EEG data. This approach enhances the robustness and accuracy of emotion and consciousness recognition across subjects, paving the way for advanced neural systems applications. 🌐

## 🛠️ Prerequisites

To get started, ensure you have the following:

- 🐍 Python 3.8 or higher
- 📦 Required Python packages :
  - NumPy
  - PyTorch
  - SciPy
  - Scikit-learn
  - Pandas
  - ......
- 📊 Datasets: SEED and SEED-IV
- 💻 A computing environment (GPU recommended for faster training ⚡)

## ⚙️ Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 📂 Dataset Preparation

This project uses the **SEED** and **SEED-IV** datasets for EEG-based emotion recognition. Follow these steps to prepare the data:

1. 📥 Download the SEED and/or SEED-IV datasets from their official sources.
2. 📂 Place the extracted dataset files (DE) in the appropriate directory.
3. ✅ Ensure the dataset files are correctly formatted and accessible for preprocessing.

## 🚀 Running the Experiments

Follow these steps to run the experiments and reproduce our results:

1. **Create a Result Directory** 📁
   In the project root directory, create an empty folder named `result` to store outputs:

   ```bash
   mkdir result
   ```

2. **Preprocess the Data** 🛠️
   Run the preprocessing script to partition the dataset into source, target, and mixed domains:

   ```bash
   python datapipe.py
   ```

   **Note**: Update the dataset paths in `datapipe.py` to match your SEED/SEED-IV data locations.

3. **Run the Main Script** ▶️
   Execute the main script to train and evaluate the model:

   ```bash
   python main.py
   ```

   The script will handle model training, evaluation, and save results to the `result` directory.

## 📈 Output

- 📜 Training logs and model checkpoints will be saved in the `./result` directory.
- 📊 Evaluation metrics (e.g., accuracy, F1-score) for emotion and consciousness recognition will be logged and saved as summary files.

## 📝 Citation

If you find our work inspiring or use this code, please cite our paper:

```bibtex
@ARTICLE{11142795,
  author={Chen, Rongtao and Xie, Chuwen and Zhang, Jiahui and You, Qi and Pan, Jiahui},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={A Progressive Multi-Domain Adaptation Network With Reinforced Self-Constructed Graphs for Cross-Subject EEG-Based Emotion and Consciousness Recognition}, 
  year={2025},
  volume={33},
  pages={3498-3510},
  keywords={Brain modeling;Emotion recognition;Electroencephalography;Accuracy;Feature extraction;Adaptation models;Probability distribution;Noise measurement;Emotional responses;Computational modeling;Electroencephalogram (EEG);emotion recognition;consciousness recognition;domain adaptation;reinforcement learning},
  doi={10.1109/TNSRE.2025.3603190}}
```

## 📬 Contact

For questions, feedback, or issues, please:

- 📧 Reach out to [chenrongtao@m.scnu.edu.cn](mailto:chenrongtao@m.scnu.edu.cn).


Thank you for exploring our work! We hope this repository sparks innovation and advances your research journey! 🌟
=======
```markdown
# 🌟 MindAligner: Progressive Multi-Domain Adaptation Network with Reinforced Self-Constructed Graphs

[![Paper](https://img.shields.io/badge/Paper-TNSRE_2025-blue)](https://doi.org/10.1109/TNSRE.2025.3603190)
[![Framework](https://img.shields.io/badge/Framework-MindSpore-blue)](https://www.mindspore.cn/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

本仓库是论文 **"A Progressive Multi-Domain Adaptation Network with Reinforced Self-Constructed Graphs for Cross-Subject EEG-Based Emotion and Consciousness Recognition"** 的官方 MindSpore 实现。该论文发表于 *IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)*, 2025。

---

## 💡 商业价值与技术泛化：从巅峰攻坚到全场景应用

本项目的核心竞争力不仅在于学术突破，更在于其**极强的算法鲁棒性**与**跨场景的商业落地潜力**：

1. **核心突破（攻克最难点）**：
   本技术率先成功应用于脑机接口领域公认的“最高难度挑战”——**意识障碍（DOC）患者的辅助诊断**。该场景下的脑电信号具有极高的非平稳性、强噪声和显著的个体差异，而我们的 `EEGAlignNet` 跨越了这些障碍，实现了高精度的状态识别。

2. **技术解耦（通用性保障）**：
   * **算法通用**：核心的渐进式域自适应（Progressive DA）与强化学习自构建图（RSCG）技术，本质上解决的是“脑电特征在不同个体间的对齐难题”，完全不依赖于特定病症。
   * **硬件兼容**：系统可无缝适配市面主流的干/湿电极脑电采集设备，大幅降低了商业化门槛。

3. **商业泛化版图**：
   基于在最困难的 DOC 领域的成功经验，该“通用对齐”技术可直接实现技术下放，快速泛化至**情绪监控、压力预警、智慧康养、专注度评估（教育/冥想）**等广阔的大众消费级与临床医疗级市场。

---

## 📂 仓库结构

```text
ICT_MindSpore/
├── main_mindspore.py      # 主训练与评估脚本 (核心逻辑)
├── Net_mindspore.py       # EEGAlignNet 架构 (包含渐进式梯度反转层)
├── RL_mindspore.py        # 基于强化学习的动态边选择模块
├── datapipe_mindspore.py  # 数据预处理与 EEG-CutMix 数据增强
├── result/                # 开源证据：训练日志与最佳权重
│   ├── best_model_cv00.ckpt   # 受试者 0 的最佳模型权重
│   └── EEGAlignNet_MS_LOG_1.csv # 实验运行详细日志
└── README.md              # 本说明文件
```

---

## 📊 数据集申请与准备

本项目在学术界标准的 **SEED** 数据集上进行了验证。

### 1. 访问与申请
由于脑电数据涉及隐私，需向官方实验室申请。
* **SEED 官网**: [BCMI SEED Dataset](https://bcmi.sjtu.edu.cn/home/seed/index.html)
* **申请方式**: 按照官网说明填写数据使用协议 (DUA)，通常需要使用机构邮箱或由导师发送申请。

### 2. 目录组织
下载并解压后，请将 `.mat` 文件放置于 `datapipe_mindspore.py` 中指定的路径下（默认示例为 `E:/EEG_data/SEED/ExtractedFeatures/`）。

---

## 🚀 环境复现步骤

### 1. 环境安装
推荐环境：Python 3.9+, MindSpore 2.2.10+。
```bash
# 示例：安装 MindSpore Ascend 版本
pip install [https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/ascend/aarch64/mindspore_ascend-2.2.10-cp39-cp39-linux_aarch64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/ascend/aarch64/mindspore_ascend-2.2.10-cp39-cp39-linux_aarch64.whl)

# 安装其他核心依赖
pip install pandas scikit-learn tqdm scipy matplotlib
```

### 2. 数据预处理
将原始 `.mat` 数据转化为 MindSpore 训练所需的 PyG 风格格式：
```bash
python datapipe_mindspore.py
```

### 3. 模型训练与验证
运行主程序，系统将自动执行 15 名受试者的 Leave-One-Subject-Out (LOSO) 交叉验证：
```bash
# 建议在服务器后台运行，以防断网中断
nohup python -u main_mindspore.py > train_progress.log 2>&1 &
```

---

## 📈 实验成果与开源验证

为了确保实验的**可复现性**并遵循开放科学原则，我们在 `result/` 目录下公开了：
* **训练日志**：包含每个受试者在训练过程中的 Loss 变化及最终准确率记录。
* **最佳模型权重**：15 个受试者分别对应的 `.ckpt` 权重文件，可直接调用进行下游特征提取或推理任务。

---

## 📝 引用

如果您在研究中使用了本项目，请引用我们的论文：

```bibtex
@ARTICLE{11142795,
  author={Chen, Rongtao and Xie, Chuwen and Zhang, Jiahui and You, Qi and Pan, Jiahui},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={A Progressive Multi-Domain Adaptation Network With Reinforced Self-Constructed Graphs for Cross-Subject EEG-Based Emotion and Consciousness Recognition}, 
  year={2025},
  volume={33},
  pages={3498-3510},
  doi={10.1109/TNSRE.2025.3603190}}
```

---

## 📬 联系我们
如有学术探讨或商业合作意向，请联系：[chenrongtao@m.scnu.edu.cn](mailto:chenrongtao@m.scnu.edu.cn)。
```

---
---

### 2. PPT 页面设计与讲稿提取 (`PPT_Design_Strategy.md`)

```markdown
# 📊 商业路演 PPT 设计方案 (单页布局设计)

**页面主题**：降维打击——从攻克核心技术壁垒到无限商业泛化

## 1. 视觉排版逻辑 (推荐采用“从左至右”的三段式演进)

### 模块一（左侧）：技术基石 (The Hardest Problem)
* **视觉元素**：放置一个简化的 `EEGAlignNet` 网络架构图，或带有剧烈噪声的复杂脑电波形图。
* **核心文案**：
  * **“攻克行业珠穆朗玛峰”**
  * 挑战脑机接口最高难度：意识障碍 (DOC) 辅助诊断。
  * 突破三大限制：强噪声、高非平稳性、极大个体差异。

### 模块二（中间）：核心桥梁 (The Universal Bridge)
* **视觉元素**：汇聚的漏斗或向右发散的渐变箭头，象征技术的转化与解耦。
* **核心文案**：
  * **“算法解耦，软硬通用”**
  * 核心算法：跨个体特征对齐（不限病种，突破个体差异）。
  * 硬件兼容：适配通用干/湿电极脑电设备（不挑硬件，成本可控）。

### 模块三（右侧）：商业泛化版图 (The Infinite Market)
* **视觉元素**：展开的网格图或蜂窝图，配以相应的场景扁平化图标（如医疗十字、笑脸、大脑、书籍等）。
* **核心文案**：
  * **临床医疗级**：DOC 促醒辅助监测、抑郁症/焦虑症客观筛查。
  * **智慧康养级**：老年人情感陪护、认知衰退（阿尔茨海默）早期预警。
  * **大众消费级**：正念冥想辅助闭环、高优人才（教育/电竞）专注度训练。

---

## 2. 核心演讲配词 (Speaker Notes)

> “各位评委/老师，大家现在看到的不仅是一项前沿的学术突破，更是一个具备庞大商业想象力的底层引擎。
> 
> 我们一开始就选择了最难的 **DOC（意识障碍）** 作为试金石。这意味着什么？这意味着如果我们的技术能够成功对齐最混乱、最嘈杂、个体差异最大的植物人脑电信号，那么这套算法就已经经历了最严苛的极限测试。
> 
> 所以，因为我们的**算法是通用的**（只做信号对齐，不挑病种），**设备也是通用的**（适配市面常规脑电帽），当我们把这套成熟的系统平移到健康人群的情绪识别、老年人的康养陪护、甚至是学生的专注度监测时，我们在技术层面完全是**降维打击**。这就是我们项目从攻克单一的医学高峰，走向全场景商业泛化的核心逻辑。”
```
>>>>>>> decab02 (feat: initial commit of MindAligner MindSpore version including logs and checkpoints)
