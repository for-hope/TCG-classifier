import os
import random


execution_path = os.getcwd()
data_path = os.path.join(execution_path, 'data')
images_path = os.path.join(data_path, 'images')


def split_data(image_dir_path, training_percentage):
    images = [os.path.join(image_dir_path, image)
              for image in os.listdir(image_dir_path)]
    data_length = len(images)
    random.shuffle(images)
    training_data_length = int((training_percentage / 100) * data_length)
    testing_data_length = data_length - training_data_length
    training_data = images[:training_data_length]
    testing_data = images[-testing_data_length:]

    dir_name = os.path.basename(os.path.normpath(image_dir_path))
    train_path = os.path.join(data_path, 'train', dir_name)
    test_path = os.path.join(data_path, 'test', dir_name)

    print("Renaming training data...")
    for image in training_data:
        image_name = os.path.basename(os.path.normpath(image))
        os.renames(image, os.path.join(train_path, image_name))

    print("Renaming testing data...")
    for image in testing_data:
        image_name = os.path.basename(os.path.normpath(image))
        os.renames(image, os.path.join(test_path, image_name))


def main():
    for image_dir in os.listdir(images_path):
        image_dir_path = os.path.join(images_path, image_dir)
        split_data(image_dir_path, 80)


if __name__ == '__main__':
    main()
