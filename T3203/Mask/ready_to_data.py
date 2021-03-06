import os
import random
from PIL import Image


# 샘플링을 할지 결정하는 클래스입니다. 이미지 디렉토리 주소와 샘플 사이즈 비율을 받습니다. 샘플 사이즈 비율이 0이면 test 데이터를 만들지 않습니다.
# 변수를 받으면 이미지 디렉토리 이름들이 모여있는 리스트를 출력합니다.
# 클래스와 함수를 선언했을뿐 동작은 데이터를 사용하는 파일로 가서 불러옵니다. test_data.py에 예시가 있습니다.
class Sampling(object):

    def __init__(self,images_path,sampling_size_rate=0):
        
        image_directory_names = [i for i in os.listdir(images_path) if '._' not in i]
        image_directory_names.sort()

        if sampling_size_rate==0:
            self.train_image_directory_names = image_directory_names
        else:
            # systematic sampling
            first_number_choise = random.randrange(0,int(1/sampling_size_rate))
            # divide original directory names with test directory names and trian directory names
            self.test_image_directory_names = [image_directory_names[i] for i in range(len(image_directory_names)) if (i+int(1/sampling_size_rate)-first_number_choise)%int(1/sampling_size_rate)==0]
            self.train_image_directory_names = [image_directory_names[i] for i in range(len(image_directory_names)) if (i+int(1/sampling_size_rate)-first_number_choise)%int(1/sampling_size_rate)!=0]


# 각종 변수들(리턴값 확인)을 출력하는 함수입니다. 이미지 디렉토리 주소와 Sampling 클래스에서 나온 이미지 디렉토리 이름 리스트를 받습니다.
def refine_data(images_path,image_directory_names:list):

    def create_image_list(image_directory_names:list):
        image_directory_path = [images_path + "/" + i + "/" for i in image_directory_names]
        image_file_path = [i + j for i in image_directory_path for j in os.listdir(i) if '._' not in j]
        image_list = [Image.open(i) for i in image_file_path]    
        return image_list, image_file_path

    def create_individual_class_lists(image_file_path:list)->list:

        splited_path = [i.split('/') for i in image_file_path]

        directory_name = []
        file_name = []

        for i in splited_path:
            directory_name.append(i[-2])
            file_name.append(i[-1])

        splited_directory_name = [i.split('_') for i in directory_name]

        gender =[]
        age = []

        for _,i,_,j in splited_directory_name:
            gender.append(i)
            age.append(float(j))

        mask = []

        for i in file_name:
            mask.append(i.split('.')[0])

        gender_dict = {
            'male': 0,
            'female': 1,
            }
        mask_dict = {
            'mask1':0,
            'mask2':0,
            'mask3':0,
            'mask4':0,
            'mask5':0,
            'incorrect_mask':1,
            'normal':2,
        }

        gender_class = []
        for i in gender:
            gender_class.append(gender_dict[i])

        mask_class = []
        for i in mask:
            mask_class.append(mask_dict[i])

        age_class = []

        def change_age_to_class(x) -> int:
            answer = 0
            if x < 30:
                answer = 0
            elif x >= 30 and x < 60:
                answer = 1
            elif x >= 60:
                answer =2
            return answer

        for i in age:
            age_class.append(change_age_to_class(i))

        return mask_class, gender_class, age_class, age

    def create_mixed_class_list(mask:list,gender:list,age:list)->list:

        mixed_class_dict = {
        (0,0,0):0,
        (0,0,1):1,
        (0,0,2):2,
        (0,1,0):3,
        (0,1,1):4,
        (0,1,2):5,
        (1,0,0):6,
        (1,0,1):7,
        (1,0,2):8,
        (1,1,0):9,
        (1,1,1):10,
        (1,1,2):11,
        (2,0,0):12,
        (2,0,1):13,
        (2,0,2):14,
        (2,1,0):15,
        (2,1,1):16,
        (2,1,2):17,
        }

        mixed_class = []
        for i,j,k in zip(mask,gender,age):
            mixed_class.append(mixed_class_dict[(i,j,k)])
        
        return mixed_class

    image_list,image_file_path = create_image_list(image_directory_names)    
    mask_class,gender_class,age_class, age = create_individual_class_lists(image_file_path)
    mixed_class = create_mixed_class_list(mask_class,gender_class,age_class)
    return image_list,mask_class,gender_class,age_class,age, mixed_class


if __name__ == '__main__':
    
    # S = Sampling(images_path,sampling_size_rate)
    # test_image_list,test_mask_class,test_gender_class,test_age_class,test_mixed_class = refine_data(S.test_image_directory_names)
    # train_image_list,train_mask_class,train_gender_class,train_age_class,train_mixed_class = refine_data(S.train_image_directory_names)

    # print(set(train_gender_class))

    pass