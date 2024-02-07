# CAER-Baselines

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

A PyTorch-powered repository designed for replicating various Computer Vision tasks in the field of CAER (Context-Aware Emotion Recognition). This project was created with [horizon](https://github.com/larrylu0426/horizon) by Larry Lu.

## CAER-Net-S with CAER-S
This is the implementation for paper [Context-Aware Emotion Recognition Networks](https://caer-dataset.github.io/file/JiyoungLee_iccv2019_CAER-Net.pdf)

### Dataset
I use [CAER-S](https://drive.google.com/a/yonsei.ac.kr/file/d/1cqB_5UmFQXacjPeRb8Aw1VE2v0vO4bdo/view?usp=sharing) and [CAERS-Annotations](https://github.com/larrylu0426/CAERS-Annotations) as the data annotations. Here is an overview of the annotations:

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
    use_gpu: true
    gpu_ids: 0
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
| Y  | Y | [67.83%](https://drive.google.com/file/d/17wX2C6DCyJlb2YpcYd2_SMTQRGPd7ftW/view?usp=sharing) |
| N  | Y | [63.07%](https://drive.google.com/file/d/1hZAJv1aaxIyK17MeENYi2wW6-QUHQhGz/view?usp=sharing) |
| N  | N | [74.31%](https://drive.google.com/file/d/1szGFn_JJxJt-gdlWhaasVW6_UkPDMTJO/view?usp=sharing) |

## Emotic with baseline model
This is the implementation for paper [Context Based Emotion Recognition using EMOTIC Dataset](https://arxiv.org/pdf/2003.13401)

### Dataset
Here is an overview of the annotations:

- Phase: train<br>
Total: 23266<br>
Counts:
 {'Affection': 1063, 'Anger': 209, 'Annoyance': 368, 'Anticipation': 5335, 'Aversion': 168, 'Confidence': 4059, 'Disapproval': 325, 'Disconnection': 1462, 'Disquietment': 479, 'Doubt/Confusion': 659, 'Embarrassment': 152, 'Engagement': 12814, 'Esteem': 851, 'Excitement': 4394, 'Fatigue': 538, 'Fear': 177, 'Happiness': 5630, 'Pain': 188, 'Peace': 1691, 'Pleasure': 2103, 'Sadness': 404, 'Sensitivity': 360, 'Suffering': 260, 'Surprise': 417, 'Sympathy': 718, 'Yearning': 668}
- Phase: val<br>
Total: 3315<br>
Counts:
 {'Affection': 661, 'Anger': 110, 'Annoyance': 278, 'Anticipation': 2958, 'Aversion': 196, 'Confidence': 1892, 'Disapproval': 229, 'Disconnection': 842, 'Disquietment': 419, 'Doubt/Confusion': 390, 'Embarrassment': 160, 'Engagement': 3133, 'Esteem': 708, 'Excitement': 1964, 'Fatigue': 235, 'Fear': 169, 'Happiness': 2276, 'Pain': 162, 'Peace': 627, 'Pleasure': 1119, 'Sadness': 175, 'Sensitivity': 144, 'Suffering': 152, 'Surprise': 300, 'Sympathy': 695, 'Yearning': 403}
- Phase: test<br>
Total: 7203<br>
Counts:
 {'Affection': 961, 'Anger': 172, 'Annoyance': 337, 'Anticipation': 3408, 'Aversion': 209, 'Confidence': 3446, 'Disapproval': 308, 'Disconnection': 1109, 'Disquietment': 893, 'Doubt/Confusion': 1041, 'Embarrassment': 133, 'Engagement': 5592, 'Esteem': 988, 'Excitement': 3469, 'Fatigue': 425, 'Fear': 230, 'Happiness': 3466, 'Pain': 140, 'Peace': 1062, 'Pleasure': 2090, 'Sadness': 369, 'Sensitivity': 205, 'Suffering': 240, 'Surprise': 436, 'Sympathy': 605, 'Yearning': 467}

 ### Model
I revise the model in this [repo](https://github.com/Tandon-A/emotic), thanks a lot for Abhishek. I also use two pre-trained ResNet18 modules as the feature extractors and doesn't fix their parameters for the higher performance.
### Training Details
hyperparameters
```
dataset:
  value:
    name: EMOTIC
    args:
      batch_size: 52
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
      step_size: 60 # close to 50
      gamma: 0.1
trainer:
  value:
    seed: 0
    use_gpu: true
    gpu_ids: 0
    epochs: 50
    early_stop: 20
```

The [result](https://drive.google.com/file/d/1HUIXc2zAR4snzBeo_WKQL9rTXC2QTu3G/view?usp=sharing): 

    test_loss      : 0.16760232572932895
    test_mean_ap   : 0.2651345729827881
    test_mean_aae  : 0.09527605772018433