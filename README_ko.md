# Conditional_Image_Generation
2024 2학기 시각지능학습 프로젝트_이미지 생성 모델


## 목차
- [프로젝트 개요](#프로젝트-개요)
- [데이터셋](#데이터셋)
- [재현성](#재현성)
- [실행 방법](#실행-방법)
- [결과](#결과)
- [참고 문헌](#참고-문헌)<br><br>

## 프로젝트 개요
이 프로젝트는 CIFAR-100 데이터셋의 20개 슈퍼클래스에 맞춰 이미지를 생성하는 조건부 이미지 생성 모델인 BigGAN을 구현한 것입니다. 본 프로젝트에서는 각 슈퍼클래스에 해당하는 이미지를 생성하기 위해 BigGAN의 구조를 활용하고, 이를 통해 다양한 시각적 특성을 가진 이미지를 생성하는 것을 목표로 합니다. 이를 통해 조건부 생성 모델의 성능을 평가하고, 각 슈퍼클래스에 대한 이미지 생성의 품질을 분석할 수 있습니다.<br><br>








## 데이터셋
CIFAR-100 데이터셋은 100개의 클래스에 걸쳐 총 60,000개의 32x32 컬러 이미지로 구성되어 있습니다. 이 데이터셋은 20개의 슈퍼클래스로 그룹화되어 있으며, 각 슈퍼클래스는 여러 개의 세부 클래스를 포함하고 있습니다.  이 프로젝트에서는 CIFAR-100 데이터를 로드하고 전처리하는 커스텀 Dataset 클래스를 사용합니다.

데이터 변환
학습에 사용되는 데이터 변환은 다음과 같습니다


```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
```

또한, 슈퍼클래스 매핑을 적용하여, 각 슈퍼클래스에 맞는 이미지를 조건부로 생성할 수 있도록 합니다.

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


## 재현성
본 프로젝트에서는 딥러닝 모델 학습 시 재현성을 보장하기 위해 난수 시드를 고정하였습니다.

이 프로젝트는 기본적으로 시드 값을 112로 설정하였으며, 이를 통해 데이터 샘플링, 가중치 초기화 등의 과정에서 일관된 학습 결과를 얻을 수 있습니다.


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

위 코드를 통해 학습 과정에서 난수로 인해 발생할 수 있는 불일치를 최소화하여 실험의 재현성을 유지할 수 있습니다.


### Random seed 수정
위의 함수에서 원하는 시드 값으로 `set_random_seed()`의 매개변수를 수정하여 사용할 수 있습니다.

시드 값을 원하는 값으로 변경하세요. 예 : set_random_seed(42)<br><br>




## 실행 방법
### 저장소를 클론하세요.
```python
git clone https://github.com/3-norm/Conditional_Image_generation.git
```
#### 파일 구조
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
### 필요한 패키지를 설치합니다. requirements.txt 파일을 이용해 아래 명령어로 설치할 수 있습니다.

```python
pip install -r requirements.txt
```

#### CIFAR-100 데이터셋은 코드 실행 시 자동으로 다운로드 및 로드됩니다.

### 코드 실행 

1. **BigGAN 모델 학습**
   - `train.ipynb` 파일을 Jupyter Notebook에서 열고, 셀을 순차적으로 실행하세요.<br>
      랜덤 시드는 기본적으로 112로 설정되어 있으며, 필요에 따라 변경할 수 있습니다.  
   
2. **점수 계산**
   - 학습이 끝난 후, 이어서 셀을 순차적으로 실행하여 세 가지 평가지표 점수를 계산하세요.

3. **생성된 이미지 샘플 확인**

   - 1000 반복마다 생성된 이미지 샘플이 samples 폴더에 저장됩니다.<br>
    이 폴더에는 슈퍼클래스 20개에 대해 각각 10장의 이미지가 생성되어 있으며, 10x20 크기로 배열되어 있습니다.<br>




<br><br>
## 결과

### 모델 주요 파라미터
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
## 참고 문헌
BigGAN : https://github.com/YangNaruto/FQ-GAN/tree/master/FQ-BigGAN <br>
CIFAR-100 dataset: [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
논문 : Brock, Andrew. "Large Scale GAN Training for High Fidelity Natural Image Synthesis." arXiv preprint arXiv:1809.11096 (2018)

