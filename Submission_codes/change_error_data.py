import os
import shutil 


# 0. 터미널에 conda install shutil을 설치해 주세요.
# 0. 해당 파일을 그대로 실행하면 백업 폴더 만들고 그 폴더에 있는 오류파일들을 수정합니다. 여기까지만 하시면 ready_to_data.py로 넘어가도 됩니다.
# 0. 추가적인 수정을 원하시면 아래부터 읽어보시고 하단에 선언된 클래스와 메소드를 수정하시면 됩니다.
try:
    shutil.copytree("/opt/ml/input", "/opt/ml/backup/input")
except:
    print("복사한 폴더가 이미 있어요.")

# 1. 이미지 디렉토리 주소 설정(이미지 디렉토리 백업을 권장)
images_path = '/opt/ml/backup/input/data/train/images'

# 2. 원하는 속성의 값을 바꿔주는 클래스
class ChangeAttr(object):

    def __init__(self,images_path): 
        # 이미지 디렉토리가 있는 주소를 받아서 이미지 디렉토리 안에 있는 사람별로 정리된 디렉토리 리스트들을 만든다.
        image_directory_names = [i for i in os.listdir(images_path) if '._' not in i]
        image_directory_names.sort()
        self.image_directory_names = image_directory_names
        

    def change_incorrect_mask_and_normal(self,mask_error_list:list): 
        # incorrect_mask.jpg파일과 normal.jpg파일의 이름을 바꾸는 메소드다. 
        # 바꾸기를 원하는 인덱스 리스트를 입력하면 실행가능하다.
        mask_error_directory_names = [i for i in self.image_directory_names if str(i[:6]) in mask_error_list]
        for i in mask_error_directory_names:

            change_path = os.path.join(images_path,i)
            a1_file = os.path.join(change_path ,'incorrect_mask.jpg')
            A1_file = os.path.join(change_path ,'A1.txt')
            a2_file = os.path.join(change_path ,'normal.jpg')

            os.rename(a1_file,A1_file)
            os.rename(a2_file,a1_file)
            os.rename(A1_file,a2_file)

            print(f"{i}'s file names have been changed.")


    def change_gender(self,gender_error_list:list):
        # gender를 바꾸는 메소드다. 
        # 바꾸기를 원하는 인덱스 리스트를 입력하면 실행가능하다.
        gender_error_directory_names = [i for i in self.image_directory_names if str(i[:6]) in gender_error_list]
        for i in gender_error_directory_names:
            before = os.path.join(images_path,i)
            print(before)
            index_gender_region_age_list = i.split('_')
            if index_gender_region_age_list[1] == 'female':
                index_gender_region_age_list[1] = 'male'
            elif index_gender_region_age_list[1] == 'male':
                index_gender_region_age_list[1] = 'female'
            i = '_'.join(index_gender_region_age_list)
            after = os.path.join(images_path,i)
            print(after)
            os.rename(before,after)

            print(f"{i}'s gender has been changed.")

    def change_age(self,age_error_list:list, age_which_would_be_changed:int):
        # 나이를 바꾸는 메소드다.
        # 바꾸기를 원하는 인덱스 리스트와 나이를 입력하면 실행 가능하다.
        # 나이는 마스크착용유무와 성별처럼 반대로 전환이 안되게 때문에 원상복귀하는 방법은 수정 전 나이를 기록했다가 적용하거나 백업파일을 초기화 해야한다.
        gender_error_directory_names = [i for i in self.image_directory_names if str(i[:6]) in age_error_list]
        for i in gender_error_directory_names:
            before = os.path.join(images_path,i)
            print(before)
            index_gender_region_age_list = i.split('_')
            index_gender_region_age_list[3] = str(age_which_would_be_changed)
            i = '_'.join(index_gender_region_age_list)
            after = os.path.join(images_path,i)
            print(after)
            os.rename(before,after)
            print(f"{i}'s age has been changed.")

# 수정 할 디렉토리의 인덱스. 사람별 디렉토리 앞에 있는 인덱스를 기준으로 하며 str이다.
mask_error_list = ['000020','004418','005227']
female_error_list = ['000225','000664','000767','001509','003113','003223','004281','006359','006360','006361','006362','006363','006364','006424','000667','000725','000736','000817','003780','006504']
male_error_list = ['001498-1','004432','005223']
age29_error_list = ['001009','001064','001637','001666','001852']
age61_error_list = ['004348']

CA = ChangeAttr(images_path)
CA.change_incorrect_mask_and_normal(mask_error_list)
CA.change_gender(female_error_list)
CA.change_gender(male_error_list)
CA.change_age(age29_error_list,29)
CA.change_age(age61_error_list,61)

