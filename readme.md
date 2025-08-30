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
@article{Chen2025progressive,
  title={A Progressive Multi-Domain Adaptation Network with Reinforced Self-Constructed Graphs for Cross-Subject EEG-Based Emotion and Consciousness Recognition},
  author={Chen, Rongtao and Xie, Chuwen and Zhang, Jiahui and You, Qi and Pan, Jiahui},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2025},
  publisher={IEEE}
}
```

## 📬 Contact

For questions, feedback, or issues, please:

- 📧 Reach out to [chenrongtao@m.scnu.edu.cn](mailto:chenrongtao@m.scnu.edu.cn).

Thank you for exploring our work! We hope this repository sparks innovation and advances your research journey! 🌟