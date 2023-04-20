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

## Auto training
```bash
python autotrain.py
```

### SVM
```bash
python train_svm.py
```

### FCN
```bash
python train/trainer.py --config="config/config_fcn_logmel_bce.yaml"
```

## Inference
```bash
python infer.py --model_folder="models/"
```

## Summary all models
```bash
python generate_results.py
```
