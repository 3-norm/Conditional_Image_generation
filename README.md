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
This project implements deep learning CNN models ShakePyramidNet and wide-resnet for image classification using the CIFAR-100 dataset.<br><br>




## Dataset
The CIFAR-100 dataset consists of a total of 60,000 32x32 color images across 100 classes. Each class contains 600 images, divided into 50,000 training images and 10,000 test images. In this project, we use a custom Dataset class to load and preprocess the CIFAR-100 data.

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

1. **Run the Wide ResNet model**
   - Open the `train.ipynb` file in Jupyter Notebook and execute the cells sequentially.  
        The random seed is set to 112 by default, but you can change it as needed.
   
2. **Run the Score Metrics**
   - 


    **Setting the Model File Paths**





<br><br>
## Results

### Model Parameters
#### BigGAN
> - **Epochs**: 500
> - **Learning Rate (LR)**: D : 0.0001 , G : 0.0002

>





#### Random Seed: 112
|   Score Metrics      | 
|---------------|
| FID    |  |
| Inception-Score   |   |
| Intra-FID  |    |



<br><br>
## References
BigGAN : https://github.com/YangNaruto/FQ-GAN/tree/master/FQ-BigGAN <br>
CIFAR-100 dataset: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
