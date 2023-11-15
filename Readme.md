# CAER-Baselines

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

A PyTorch-powered repository designed for replicating various Computer Vision tasks in the field of CAER (Context-Aware Emotion Recognition). This project was created with [horizon](https://github.com/larrylu0426/horizon) by Larry Lu.

## CAER-Net-S with CAER-S
This is the implementation for paper [Context-Aware Emotion Recognition Networks](https://caer-dataset.github.io/file/JiyoungLee_iccv2019_CAER-Net.pdf)

### Dataset
I use [CAER-S](https://drive.google.com/a/yonsei.ac.kr/file/d/1cqB_5UmFQXacjPeRb8Aw1VE2v0vO4bdo/view?usp=sharing) and [CAERS-Annotations](https://github.com/larrylu0426/CAERS-Annotations) as the data annations. Here is an overview of the annotations:

- Phase: train<br>
Total: 47966<br>
Counts:  {'Angry': 6856, 'Disgust': 6890, 'Fear': 6873, 'Happy': 6814, 'Neutral': 6799, 'Sad': 6910, 'Surprise': 6824}
- Phase: test<br>
Total: 20868<br>
Counts:  {'Angry': 2988, 'Disgust': 2980, 'Fear': 2972, 'Happy': 2990, 'Neutral': 2985, 'Sad': 2983, 'Surprise': 2970}

### Model
I use the model without any changes in this [repo](https://github.com/ndkhanh360/CAER), thanks a lot for Khanh.

### Training Details
hyperparameters
```
dataset:
  value:
    name: CAERS
    args:
      batch_size: 32
      num_workers: 16
optimizer:
  value:
    name: SGD
    args:
      lr: 5.0e-3
      momentum: 0.9
      nesterov: true
lr_scheduler:
  value:
    name: StepLR
    args:
      step_size: 160 # close to 150
      gamma: 0.1
trainer:
  value:
    seed: 0
    n_gpu: 1
    epochs: 150
    early_stop: 20
```

The StepLR  outlined in the original paper (`dropped by a factor of 10 every 4 epochs`) has not been embraced due to its adverse impact on performance—instead of improvement, a decline is observed.  Personally, I find that, for a batch size of 32, a learning rate of 5.0e-3 is sufficiently small.  The incorporation of every 4 epochs in the learning rate reduction leads to a rapid drop, making it challenging to enhance performance.

The data augmentation techniques employed in the original paper (`data augmentation schemes such as flips, contrast, and color changes`) are ambiguous. The network utilizes two types of images: face and context. However, the authors do not explicitly state whether these augmentation methods apply to one or both image types. Based on my experimentation, introducing data augmentation to face images alone can enhance performance, achieving up to 84.78%. Nevertheless, the attention maps displayed exhibit peculiar characteristics. In light of these maps, it is inferred that the contextual features may not have been adequately learned.
![Alt text](<docs/pic1.png>)
Here are the additional results and model links:
|  face   | context | performace|
|  ----  | ----  | ----  |
| Y  | N | [84.78%](https://drive.google.com/file/d/1vTJZoZJaofVThOlUQQ8yOQ0YHEizSvCL/view?usp=sharing) |
| Y  | Y | [67.19%](https://drive.google.com/file/d/1L4ro26poL5Jg6KQdE1KIt4xvE8RpjaZp/view?usp=sharing) |
| N  | Y | [62.84%](https://drive.google.com/file/d/1Z1wdBtWn5Z2BSNGFRmNOJk6GwrMy3K0v/view?usp=sharing) |
| N  | N | [74.31%](https://drive.google.com/file/d/1szGFn_JJxJt-gdlWhaasVW6_UkPDMTJO/view?usp=sharing) |