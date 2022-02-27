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

## Augmentation, transform


## Imbalanced, Loss
1. Label Smoothing
    - https://3months.tistory.com/465

2. TTA
     - https://www.kaggle.com/luyujia/pytorch-my-tta-function-easy-to-understand

## Ensemble
1. Kfold 사용 (데이터로더와 함께)
    - https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
