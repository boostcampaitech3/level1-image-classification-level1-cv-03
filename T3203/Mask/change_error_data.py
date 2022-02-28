import os

# directory by person
images_path = '/opt/ml/BackUp/input/data/train/images'
image_directory_names = [i for i in os.listdir(images_path) if '._' not in i]
image_directory_names.sort()

mask_error_list = ['000020','004418','005227']
female_error_list = ['000225','000664','000767','001509','003113','003223','004281','006359','006360','006361','006362','006363','006364','006424','000667','000725','000736','000817','003780','006504']
male_error_list = ['001498-1','004432','005223']
age29_error_list = ['001009','001064','001637','001666','001852']
age61_error_list = ['004348']

class ChangeError(object):
    def __init__(self,image_directory_names):
        self.image_directory_names = image_directory_names
        

    def change_incorrect_mask_and_normal(self,mask_error_list:list):

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


    def change_gender(gender_error_list:list):

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

    def change_age(age_error_list:list, age_which_would_be_changed:int):

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

