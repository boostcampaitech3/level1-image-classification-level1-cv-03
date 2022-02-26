### 백경륜 T3101 코드 구현 폴더

1. Augmentation
    - Albumentation : torchvision transform 라이브러리보다 빠르고 다양한 Augmentation이 가능
    - Cutmix : 랜덤한 이미지들끼리 잘라서 붙이는 기법. 이미지 패치 사이즈를 랜덤한 값에서 절반(1/2)으로 변경
 
2. Imbalanced Problem
    - Focal Loss 
    - WeightedRandomSampler, ImbalancedDatasetSampler

3. Modeling
    - Model_Selection : Resnet, EfficientNEt, ViT .. 등 Pretrained Model를 사용
    - Multi Label Classifier : Conv Layer 이후 task별(mask, gender, age) FC Layer 브랜치를 만들어서 따로 예측

