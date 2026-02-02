# 使用说明

## 项目结构

```
NER_models/
├── data/
│   └── msra/
│       ├── train.txt
│       └── test.txt
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── bilstm_crf.py
│   ├── bert_crf.py
│   ├── can_ner.py
│   ├── lattice_lstm.py
│   ├── w2ner.py
│   ├── flat.py
│   ├── softlexicon.py
│   ├── lebert.py
│   ├── mect.py
│   └── zen.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py # didn't implement
├── configs/
│   └── config.py
├── main.py
└── requirements.txt

```

## 安装依赖:

```
pip install -r requirements.txt
```

## 准备数据:

将MSRA数据集放在data/msra/目录下

## 运行基准测试:

```
python main.py
```

