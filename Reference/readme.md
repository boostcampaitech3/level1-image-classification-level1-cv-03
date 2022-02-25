## 모델링, 성능 개선 아이디어 및 레퍼런스

### 모델 구조

1. 하나의 모델에서 여러 브랜치를 만든다.
    - conv layer에서 빠져나온 후, 마스크/성별/나이대 별로 새로운 브랜치로 학습을 한다.
    - 레퍼런스
        - https://discuss.pytorch.org/t/how-to-train-the-network-with-multiple-branches/2152 : forward에서 3개로 분리? -> return 에서 값 3개 나오고 각 값별로 loss과 optim 따로?
        - https://discuss.pytorch.org/t/how-to-train-multi-branch-output-network/95199  

  

2. 아예 3가지 Conv 모델로 돌아가서 마지막에 3개의 flatten을 concat 한다.
    - https://stackoverflow.com/questions/66786787/pytorch-multiple-branches-of-a-model
