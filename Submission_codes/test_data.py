from ready_to_data import Sampling, refine_data


images_path = '/opt/ml/backup/input/data/train/images'
sampling_size_rate = 0.2

# 클래스와 함수가 받는 변수는 이미지 디렉토리 주소와 샘플링 사이즈 비율입니다.

# 아래와 같이 샘플링을하고 데이터를 불러올 수 있습니다.
# 샘플링 사이즈를 0으로 설정하면 test_image_directory_names는 만들어지지 않습니다.
S = Sampling(images_path,sampling_size_rate)
test_image_list,test_mask_class,test_gender_class,test_age_class,test_mixed_class = refine_data(images_path,S.test_image_directory_names)
train_image_list,train_mask_class,train_gender_class,train_age_class,train_mixed_class = refine_data(images_path,S.train_image_directory_names)

# 간단히 몇가지 출력해보면
print(set(test_mask_class))
print(test_image_list[:5])
print(set(train_mixed_class))
# 이런식으로 불러올 수 있습니다.