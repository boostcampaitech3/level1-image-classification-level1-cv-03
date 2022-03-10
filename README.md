## 마스크 착용 상태 분류 대회 Wrap Up 

### 프로젝트 개요
#### 목적: 마스크 착용 상태 분류
- 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task
- 사람 얼굴 이미지를 학습하는 모델을 생성하여 성별, 마스크 착용 여부, 나이를 기준으로 18개의 클래스로 분류한다.

#### 활용 장비 및 재료
- 개발 환경 : Aistages server, VScode, Jupyter NoteBook
- 협업 Tools : Git, GitHub, Wandb, TensorBoard, Notion
  - 깃허브 Reference 폴더 : [아이디어 및 모델 성능 개선 방법 레퍼런스 공유](https://github.com/boostcampaitech3/level1-image-classification-level1-cv-03/tree/main/Reference)
  - 노션 프로젝트 관리 폴더 : [현재 작업 현황과 실험 결과 공유](https://www.notion.so/Level-1-P-stage-dc32dfcef44847ef987c4cac491e00d1)

- Dataset
  - 입력값 : train data 2,700명 x 사진 7장 + evl data 12,600명 x 사진 1장 = 31,500jpg 파일
  - 출력값 : 마스크 착용 여부(Wear / Incorrect / NotWear) [3 class],
  - 성별(Male / Female) [2 class],
  - 나이(~30, 30~60,60~) [3 class]
  - 마스크 착용 여부 x 성별 x 나이 = [18 class]

#### 프로젝트 팀 구성 및 역할
- 김대유 : baseline 기반 Wandb 추가 2 & 3 Multi Lable Model 설계 및 학습
- 이융희 : baseline 코드 리뷰 및 리팩토링 & EfficientNet Model Custom 및 hyper parameter 튜닝
- 정효재 : 데이터 탐색 후 수정 및 인사이트 도출
- 이상진 : 다양한 pretrained 모델 적용 및 테스트, github repository 구성
- 백경륜 : Augmentation 기법 구현 및 Model 설계, 팀 프로젝트 관리 템플릿 작성

#### 프로젝트 수행 절차
![project](https://user-images.githubusercontent.com/48708496/157568880-ff6cfd5c-5e6f-4598-9fab-09ae7f010e21.jpg)


### 자체 평가
#### 계획 대비 달성도
- Base code 사용 미숙 및 협력과 거리가 먼 협업으로 인하여 계획한 f1 스코어와 등수는 달성하지 못하였다.
- 프로젝트를 완수하고 본래의 목적인 마스크 착용 유무를 구별하는 기능은 충분히 구현됨에 따라 프로젝트 자체에
대한 기대치는 충족하였다.
#### 잘한 점
- 그래도 프로젝트의 목적인 마스크 착용 유무를 구별하는 기능은 상용화 할 수 있는 수준으로 보인다.
- 프로젝트를 끝까지 진행하고 의도한 모델을 생성할 수 있는 만큼 의미있는 시간이었다고 생각한다.
- 시도했으나 잘 되지 않았던 것들
- 부족한 dataset을 Augmentation으로 어느정도 성능을 끌어올릴 수 있었지만 age와 같은 속성 자체의 근본적인
문제까지 해결할 수는 없었다.
- 마스크 유무를 먼저 파악하고 그에 따른 나이와 성별을 파악하는 모델을 만들었으나, Conv를 이미 18개 클래스로
나눴던 모델을 가져온거라 마스크의 유무의 상관없이 얼굴 위쪽의 Feather를 가져오는 듯 했다. 그래서 마스크가
있는 것과 없는 데이터를 따로 학습하면 더 좋은 성능이 나올거라 기대한다.
- 모델 평가에서 점수가 제일 잘 나온 두 가지 모델을 ensemble시키는 과정에서의 정확도 향상을 달성하지 못하였다.
#### 아쉬운 점
- P Stage임에도 불구하고 U Stage와 같이 개인 학습만 몰두하다보니 커뮤니케이션이 잘 안돼서 협업을 못했다.
- 협업이 안되니까 작업효율이 좋지 못해서 다른 창의적인 시도를 하기 어려웠다.
#### 배운 점
- 개인이 혼자 진행하는 실험과 프로젝트가 아닌 5명이 팀으로서 같은 코드를 가지고 다양한 협업툴을 통하여 함께
작업하는 법을 배울 수 있었다.
- 딥러닝 프로젝트가 진행되는 일련의 절차와 대략적인 구조를 배울 수 있었다.
- 모델의 성능을 향상시키기 위하여, 원래 존재하는 모델을 수정하는 것이 아닌 데이터 증강과 수정을 통하여 모델의
성능을 극적으로 향상시킬 수 있다.
- Cutmix, multi label, albumentation 등 다양한 데이터 증강 방법과 기술을 배울 수 있었다


---
### Team Notice
aistages 서버에 git 연동하기

0. git은 이미 설치가 됌
1. gh 설치하기(conda install gh --channel conda-forge)
2. 자신의 깃 허브와 연동하기(gh auth login)
3. 부스트캠프 안에 있는 레포지토리 클로닝(gh repo clone boostcampaitech3/level1-image-classification-level1-cv-03)
4. add, commit, push 작동 확인하기(오류나면 pull해보고 다시 하기)  

<br />

* 대회 첫째 주 까지는 각자 코드를 이해하고 돌려보고 제출까지 하는 과정에서 협업이 필요하지 않기 때문에 공용 폴더에 ```push```하지 않아도 될 것 같습니다. 
* 2/24(목) 오피스아워 시간에 베이스코드 해설을 해주시니 개인적으로 공부하고 코드를 돌려보는 것이 아니면, 공용 폴더에 있는 코드를 수정 및 ```push```해주시고, 해당 코드로 돌려서 나온 ```submission.csv``` 파일을 제출해주시면 될 것 같습니다.
* 공용 폴더에 있는 파일을 ```push``` 하실 때는 아마 두 가지 상황이 있을 수 있는데
  1. 단순히 코드 수정을 하고 ```push```할 때는 ```commit message```에 무엇을 어떻게 수정 하였는지 간단하게 설명을 적어주시면 되고
  2. 코드 수정 및 대회 페이지에 제출을 한 경우에는 ```commit message```에 수정 내용과 ```YYYY-MM-DD_Submission_7``` 과 같이 제출 기록도 함께 표시해주시면 좋을 것 같습니다.
  




  



