import os

import torchvision
from PIL import Image
from torchvision import transforms
from util.attack_util import filename_from_dir


def set_tensor_image_list(path):
    file_list = filename_from_dir(path)
    image_list = []

    for file in file_list:
        img = Image.open(os.path.join((path + "/" + file)))
        img = transforms.ToTensor()(img)
        image_list.append(img)

    return image_list


def tensor_image_save(instance, path, name):
    torchvision.utils.save_image(instance, os.path.join(path, name))


def aug_image_save(aug, instance, path, name):
    img = aug(instance)
    tensor_image_save(img, path, name)


def main():

    flip = transforms.RandomHorizontalFlip(p=1)
    perspective = transforms.RandomPerspective()
    rotate = transforms.RandomRotation(20, expand=False)

    brightness_jitter = transforms.ColorJitter(brightness=(0.2, 3))
    contrast_jitter = transforms.ColorJitter(contrast=(0.2, 3))
    saturation_jitter = transforms.ColorJitter(saturation=(0.2, 3))
    blur = transforms.GaussianBlur(kernel_size=(7, 13))

    random_choice = transforms.RandomChoice([flip,
                                             perspective,
                                             rotate,
                                             brightness_jitter,
                                             contrast_jitter,
                                             saturation_jitter,
                                             blur,
                                             ])

    # # 특정 directory에만 적용 시
    # path = "C:/Users/KIM/PycharmProjects/Backdoor_Final/dataset/Augmented_Attack_Source/Single_target_trigger"
    # idx = 0
    #
    # image_list = set_tensor_image_list(path)
    # for i in range(4):
    #     for img in image_list:
    #         idx += 1
    #         aug_image_save(random_choice, img, path, f"random_augmentation_{idx}.png")

    # Base directory 하위 directory 일괄 적용 시
    base_path = "/dataset/Augmented_Training"
    folder_list = filename_from_dir(base_path)

    for folder in folder_list:
        path = os.path.join(base_path, folder)
        idx = 0
        image_list = set_tensor_image_list(path)

        for img in image_list:
            idx += 1
            aug_image_save(random_choice, img, path, f"random_augmentation_{idx}.png")


if __name__ == '__main__':

    main()