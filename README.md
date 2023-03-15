# MusicTagging (WIP)

## Introduction
General machine learning / deep learning models for music tagging.

Dataset: MagnaTagATune

- [x] SVM
- [x] FCN
- [x] ShortChunk CNN
- [ ] RNN
- [ ] TCN
- [ ] TSN
- [x] CRNN
- [ ] Attention

## Create conda environment
```bash
conda create --name tag python=3.8
conda activate tag
pip install -r requirements.txt 
```
## Preprocess
```bash
python preprocess.py
```
## Training
### SVM
```bash
python train_svm.py
```

### DNN
```bash
python train/trainer.py
```

