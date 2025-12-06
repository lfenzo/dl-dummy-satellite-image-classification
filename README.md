# Dummy Satellite Image Classification

![](./images/training_samples.png)

This project serves as a “hello world” example in computer vision, focusing on a simple image
classification task using a custom ResNet architecture to classify satellite images.

- **Problem domain:** Multi-class image classification
- **Data type:** Remote sensing satellite images

## Dataset

The data used in this project consist of low-resolution Remote Sensing (RS) images stored as `.jpg`
files. Each image is associated with exactly one of the following four classes:
- green area
- cloudy
- desert
- water

The data can be freely downloaded from the original [Kaggle Dataset](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification?resource=download).

## Performance

```
              precision    recall  f1-score   support

           0     1.0000    0.9942    0.9971       172
           1     0.9909    1.0000    0.9954       109
           2     0.9930    0.9930    0.9930       143
           3     0.9929    0.9929    0.9929       140

    accuracy                         0.9947       564
   macro avg     0.9942    0.9950    0.9946       564
weighted avg     0.9947    0.9947    0.9947       564
```

## Training Setup

| Training aspect       | Details                                         |
|------------------------|--------------------------------------------------|
| Model architecture     | Custom ResNet (trained from scratch, no transfer learning) |
| Splits                 | Stratified Holdout (80%, 10%, 10%)              |
| Epochs                 | 40                                              |
| Batch size             | 16                                              |
| Optimizer              | Adam                                            |
| LR scheduler           | OneCycleLR (max_lr = 0.1)                       |
| Gradient clipping      | max_norm = 0.1                                  |
| Loss function          | CrossEntropyLoss                                |
| Data augmentation      | Random Horizontal Flip                          |
