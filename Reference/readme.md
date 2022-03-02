# 모델링, 성능 개선 아이디어 및 레퍼런스

## Modeling

1. Multi-Branch-Model
    - conv layer에서 빠져나온 후, 마스크/성별/나이대 별로 새로운 브랜치로 학습을 한다. `경륜`
    - https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152 `경륜`
    - https://discuss.pytorch.org/t/how-to-train-multi-branch-output-network/95199 `경륜`

  
2. 아예 3가지 Conv 모델로 학습 후 마지막에 3개의 flatten을 concat 한다.
    - https://stackoverflow.com/questions/66786787/pytorch-multiple-branches-of-a-model `경륜`

3. Multi Label Classifier 또는 Multi Head Classifier 키워드로 구글링하면 많은 자료가 나옵니다.
    - https://learnopencv.com/multi-label-image-classification-with-pytorch/ `경륜`
    - 

4. Multi sample dropout : 이것도 3번과 같은 구조 같습니다.
    - https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961 `경륜`

## Train Dataset
1. 마스크에는 큰 상관이 없지만, 나이와 성별에는 Augmentation이 overfit을 유발하는 것 같음. 똑같은 사람에 대해서 이미 7장이 있는 상황이기 때문에 이미 Augment된 상태라고 봐도 무방하지 않나? `경륜`
2. 모델이 예측을 잘 못하는 50대 후반과 60대 초반 구간의 데이터를 아예 삭제해보는 것은 어떨까

## Augmentation, transform
1. 밝기와 채도에 영향이 큰 것 같음. 다른 인물이어도 같은 배경에서 찍힌 케이스들이 꽤 있기 때문
2. 기존 데이터셋에 Albumentation을 적용해서 클래스간에 불균형을 맞춘 상태에서 학습을 할 예정 `경륜`

## Imbalanced, Loss
1. Label Smoothing
    - https://3months.tistory.com/465

2. TTA
     - https://www.kaggle.com/luyujia/pytorch-my-tta-function-easy-to-understand

## Ensemble
1. Kfold 사용 (데이터로더와 함께)
    - https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
