<div align="center">
  <a href="README.md">English</a> | <a href="README_ko.md">한국어</a>
</div>
<br><br>
<div align="center">
  <img src="https://github.com/user-attachments/assets/fab970c8-1903-4e08-bfdf-098c4d43f48c" alt="image"width="60%">
</div>



## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Reproducibility](#reproducibility)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)<br><br>

## Project Overview
This project implements BigGAN, a conditional image generation model that generates images according to the 20 superclasses of the CIFAR-100 dataset. In this project, we utilize the structure of BigGAN to generate images corresponding to each superclass, aiming to create images with various visual characteristics. Through this, we can evaluate the performance of the conditional generation model and analyze the quality of image generation for each superclass.<br><br>




## Dataset
The CIFAR-100 dataset consists of a total of 60,000 32x32 color images across 100 classes. This dataset is grouped into 20 superclasses, with each superclass containing several detailed classes. In this project, we use a custom Dataset class to load and preprocess the CIFAR-100 data.

Data Transformations
The data transformations used for training are as follows:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
```
Additionally, by applying superclass mapping, it is possible to conditionally generate images that correspond to each superclass.

``` python
superclass_mapping = {
      4: 0, 30: 0, 55: 0, 72: 0, 95: 0,           # aquatic mammals
      1: 1, 32: 1, 67: 1, 73: 1, 91: 1,           # fish
      54: 2, 62: 2, 70: 2, 82: 2, 92: 2,          # flowers
      9: 3, 10: 3, 16: 3, 28: 3, 61: 3,           # food containers
      0: 4, 51: 4, 53: 4, 57: 4, 83: 4,           # fruit and vegetables
      22: 5, 39: 5, 40: 5, 86: 5, 87: 5,          # household electrical devices
      5: 6, 20: 6, 25: 6, 84: 6, 94: 6,           # household furniture
      6: 7, 7: 7, 14: 7, 18: 7, 24: 7,            # insects
      3: 8, 42: 8, 43: 8, 88: 8, 97: 8,           # large carnivores 
      12: 9, 17: 9, 37: 9, 68: 9, 76: 9,          # large man-made outdoor things
      23: 10, 33: 10, 49: 10, 60: 10, 71: 10,     # large natural outdoor scenes
      15: 11, 19: 11, 21: 11, 31: 11, 38: 11,     # large omnivores and herbivores
      34: 12, 63: 12, 64: 12, 66: 12, 75: 12,     # medium-sized mammals
      26: 13, 45: 13, 77: 13, 79: 13, 99: 13,     # non-insect invertebrates
      2: 14, 11: 14, 35: 14, 46: 14, 98: 14,      # people
      27: 15, 29: 15, 44: 15, 78: 15, 93: 15,     # reptiles
      36: 16, 50: 16, 65: 16, 74: 16, 80: 16,     # small mammals
      47: 17, 52: 17, 56: 17, 59: 17, 96: 17,     # trees 
      8: 18, 13: 18, 48: 18, 58: 18, 90: 18,      # vehicles 1
      41: 19, 69: 19, 81: 19, 85: 19, 89: 19      # vehicles 2
      }

train_loader.dataset.targets = [superclass_mapping[label] for label in train_loader.dataset.targets]

```
 



## Reproducibility
To ensure reproducibility during model training in this project, we fixed the random seed.

By default, the seed value is set to 112, which allows us to obtain consistent training results in processes such as data sampling and weight initialization.


```python
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(112)
```

By minimizing inconsistencies caused by random numbers during the training process, we can maintain the reproducibility of the experiments.


### Modifying the Random Seed
In the above function, you can modify the parameter of `set_random_seed()` to the desired seed value.

Change the seed value to the desired value. Example: set_random_seed(42)<br><br>


## How to Run
### Clone the repository.
```python
git clone https://github.com/3-norm/Conditional_Image_Generation.git
```
#### File Structure
``` 
├── .gitignore            
├── samples         
├── weights      
├── README.md              
├── README_ko.md           
├── BigGAN.py  
├── inceptionID.py      
├── layers.py
├── train.ipynb
├── train_fns.py
└── utils.py
```
### Install the required packages. You can use the following command to install using the requirements.txt file.

```python
pip install -r requirements.txt
```

#### The CIFAR-100 dataset will be automatically downloaded and loaded when running the code.

### How to Train the Models

1. **Training the BigGAN Model**
   - Open the `train.ipynb` file in Jupyter Notebook and execute the cells sequentially.<br>
     The random seed is set to 112 by default, but you can change it as needed.  

2. **Score Calculation**
   - After the training is complete, continue to execute the cells sequentially to calculate the scores for the three evaluation metrics.

3. **Check Generated Image Samples**
   - Image samples generated every 1000 iterations are saved in the samples folder.<br>
     This folder contains 10 images generated for each of the 20 superclasses, arranged in a 10x20 grid.





<br><br>
## Results

### Model Main Parameters
#### BigGAN
> - **Epochs**: 700
> - **Learning Rate (LR)**: G : 0.0002 , D : 0.0001
> - **Initialization** : G : 'N02', D: 'N02'
> - **EMA** : True




#### Random Seed: 112
|   Score Metrics      | 
|---------------|
| FID    | 9.66 |
| Inception-Score   |  36.04 |
| Intra-FID  |  8.62  |



<br><br>
## References
BigGAN : https://github.com/YangNaruto/FQ-GAN/tree/master/FQ-BigGAN <br>
CIFAR-100 dataset: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
paper : Brock, Andrew. "Large Scale GAN Training for High Fidelity Natural Image Synthesis." arXiv preprint arXiv:1809.11096 (2018)
