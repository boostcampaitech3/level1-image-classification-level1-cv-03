import os
import random
from PIL import Image


# directory by person
images_path = '/opt/ml/BackUp/input/data/train/images'
image_directory_names = [i for i in os.listdir(images_path) if '._' not in i]
image_directory_names.sort()

# systematic sampling
sampling_size_rate = 0.2
first_number_choise = random.randrange(0,int(1/sampling_size_rate))

# divide original directory names with test directory names and trian directory names
test_image_directory_names = [image_directory_names[i] for i in range(len(image_directory_names)) if (i+int(1/sampling_size_rate)-first_number_choise)%int(1/sampling_size_rate)==0]
train_image_directory_names = [image_directory_names[i] for i in range(len(image_directory_names)) if (i+int(1/sampling_size_rate)-first_number_choise)%int(1/sampling_size_rate)!=0]

def create_image_list(image_directory_names:list) -> list:
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
        age.append(int(j))

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

    def change_int_to_class(x:int) -> int:
        answer = 0
        if x < 30:
            answer = 0
        elif x >= 30 and x < 60:
            answer = 1
        elif x >= 60:
            answer =2
        return answer

    for i in age:
        age_class.append(change_int_to_class(i))

    return mask_class, gender_class, age_class

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

def refine_data(image_directory_names:list)->list:
    image_list,image_file_path = create_image_list(image_directory_names)    
    mask_class,gender_class,age_class = create_individual_class_lists(image_file_path)
    mixed_class = create_mixed_class_list(mask_class,gender_class,age_class)
    return image_list,mask_class,gender_class,age_class,mixed_class

test_image_list,test_mask_class,test_gender_class,test_age_class,test_mixed_class = refine_data(test_image_directory_names)
train_image_list,train_mask_class,train_gender_class,train_age_class,train_mixed_class = refine_data(train_image_directory_names)

class RefineData(object):
    def __init__(self):
        self.test_image_list = test_image_list
        self.test_mask_class = test_mask_class
        self.test_gender_class = test_gender_class
        self.test_age_class = test_age_class
        self.test_mixed_class = test_mixed_class
        self.train_image_list = train_image_list
        self.train_mask_class = train_mask_class
        self.train_gender_class = train_gender_class
        self.train_age_class = train_age_class
        self.train_mixed_class = train_mixed_class



if __name__ == '__main__':
    # print(first_number_choise) # check random number
    # print(len(image_directory_names)) # check original data size
    # print(len(test_image_directory_names)) # check test sampling size
    # print(test_image_directory_names[0],image_directory_names[first_number_choise]) # check sampling 1
    # print(test_image_directory_names[2],image_directory_names[first_number_choise+2*5]) # check sampling 2
    # print(len(train_image_directory_names)) # check train sampling size
    # print(set(test_image_directory_names) & set(train_image_directory_names)) # check intersection between train with test
    # print(test_image_directory_path[:2])
    # print(len(test_image_file_path) == len(test_image_directory_path)*7)
    # print(len(train_image_file_path) == len(train_image_directory_path)*7)
    # print(test_image_list[0])
    # print(train_image_list[0])
    # print(test_image_list[0],set(test_mask_class),set(test_gender_class),set(test_age_class),set(test_mixed_class))
    pass