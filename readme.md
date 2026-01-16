# 安装依赖

## 安装所有依赖
pip install -r requirements.txt
## 或者单独安装
pip install torch>=1.8.0
pip install numpy
pip install scikit-learn
pip install tqdm
pip install pytorch-crf
pip install transformers

# 训练BiLSTM-CRF
python train.py --model bilstm-crf

# 评估模型
python evaluate.py
