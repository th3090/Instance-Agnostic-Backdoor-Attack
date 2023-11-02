import os
import time
# from typing import List, Tuple, Any

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torchvision import datasets

from util.dataloader import CustomImageFolder
from util.image import img_read, img_to_tensor


def augmentation_function():
    from torchvision.transforms import transforms

    flip = transforms.RandomHorizontalFlip(p=1)
    perspective = transforms.RandomPerspective()
    rotate = transforms.RandomRotation(20, expand=False)

    brightness_jitter = transforms.ColorJitter(brightness=(0.2, 3))
    contrast_jitter = transforms.ColorJitter(contrast=(0.2, 3))
    saturation_jitter = transforms.ColorJitter(saturation=(0.2, 3))
    blur = transforms.GaussianBlur(kernel_size=(7, 13))

    augmentation_function = transforms.RandomChoice([flip,
                                                     perspective,
                                                     rotate,
                                                     brightness_jitter,
                                                     contrast_jitter,
                                                     saturation_jitter,
                                                     blur,
                                                     ])

    return augmentation_function


# TODO: How to use concat dataset
def concat_dataset(*arg):
    dataset_list = [arg]
    concated_dataset = torch.utils.data.ConcatDataset(dataset_list)
    return concated_dataset


def set_dataset(data_dir, img_size=(224, 224),
                normalization=False, augmentation=False,
                remove_list=None):

    from torchvision.transforms import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if normalization:
        transforms = transforms.Compose({
            transforms.ToTensor(),
            transforms.Resize(img_size),
            normalize,
        })

    elif augmentation:
        aug_function = augmentation_function()
        transforms = transforms.Compose({
            transforms.ToTensor(),
            transforms.Resize(img_size),
            aug_function,
        })

    else:
        transforms = transforms.Compose({
            transforms.ToTensor(),
            transforms.Resize(img_size),
        })

    if remove_list:
        dataset = CustomImageFolder(data_dir, transform=transforms, remove_list=remove_list)
    else:
        dataset = CustomImageFolder(data_dir, transform=transforms)

    return dataset


def set_data_loader(data_dir, img_size=(224, 224), workers=0, batch_size=16,
                    train=True, normalization=False, augmentation=False,
                    remove_list=None) -> DataLoader:

    if augmentation:
        ori_dataset = set_dataset(data_dir=data_dir, img_size=img_size,
                                  normalization=normalization, augmentation=False,
                                  remove_list=remove_list)
        aug_dataset = set_dataset(data_dir=data_dir, img_size=img_size,
                                  normalization=normalization, augmentation=True,
                                  remove_list=remove_list)
        dataset = concat_dataset(ori_dataset, aug_dataset)
        print("Dataset augmentation complete!")

    else:
        dataset = set_dataset(data_dir=data_dir, img_size=img_size,
                              normalization=normalization, augmentation=False,
                              remove_list=remove_list)

    if train:
        data_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True)

    else:
        data_loader = DataLoader(
            dataset,
            num_workers=workers,
            batch_size=batch_size)

    return data_loader


def make_single_instance(file_path, img_size, normalization=False):
    # return shape (1, 3, 224, 224)
    image = img_read(file_path)
    instance = img_to_tensor(image, img_size, normalization=normalization)
    return instance.view(1, *instance.shape)


def attack_test(model, img_size, device, test_dir, test_list, target_label,
                log_path, acc_label=None, normalization=False):
    print("Backdoor attack test!!!")
    asr_list = []
    acc_list = []
    # total_pred_list = []

    with torch.no_grad():
        model.eval()

        for index in range(len(test_list)):
            poisoned_count = 0
            corrected_count = 0
            file_count = 0
            pred_list = []

            attack_test_path = os.path.join(test_dir + "/" + test_list[index])

            for file_name in os.listdir(attack_test_path):
                file_count += 1
                file_path = os.path.join(attack_test_path, file_name)
                instance = make_single_instance(file_path, img_size, normalization=normalization)
                instance = instance.to(device)
                pred = model(instance)

                # pred_list.append(pred.argmax().data.detach().cpu().numpy())
                pred_list.append(pred.data.detach().cpu().numpy().argmax())
                if pred.argmax() == target_label:
                    poisoned_count += 1

                if acc_label is not None and pred.argmax() == acc_label[index]:
                    corrected_count += 1

            acc = (corrected_count / file_count * 100)
            asr = (poisoned_count / file_count * 100)
            # print(f"{test_list[index]}_Attack Success Rate = {asr}%")
            acc_list.append(acc)
            asr_list.append(asr)
            # total_pred_list.append(pred_list)

    # print(total_pred_list)
    print(f"ACC_List = {acc_list}")
    print(f"ASR_List = {asr_list}")
    print('-' * 64)
    write_asr(log_path, asr_list)


# get current learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_loader, learning_rate, device, batch_size=32, num_epochs=10,
          attack=False, img_size=None, attack_test_dir=None, attack_test_list=None,
          attack_target_label=None, attack_log_path=None, acc_label=None,
          normalization=False):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    loss_history = {'train': []}
    metric_history = {'train': []}

    # best_loss = float('inf')
    # best_model_wts = copy.deepcopy(model.state_dict())

    # Training
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        # current_lr = get_lr(optimizer)
        model.train()

        train_running_loss = 0.0
        train_correct_pred = 0.0
        train_num_batch = 0.0

        for index, data in enumerate(train_loader):
            train_num_batch = index
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            _, pred = torch.max(output, 1)
            train_correct_pred += (pred == label).sum()

            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += float(loss.item())

        train_loss = train_running_loss / (train_num_batch + 1)
        train_accuracy = (train_correct_pred.item() / (batch_size * (train_num_batch + 1)) * 100)

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_accuracy)

        end_time = time.perf_counter()

        print('epoch {}/{}'.format(epoch + 1, num_epochs), end=" ")
        print('train loss: {:.6f}, train accuracy: {:.2f}%, time: {:.4f}s'.format(train_loss,
                                                                                  train_accuracy,
                                                                                  (end_time - start_time)))
        # print('-' * 10)

        if attack:
            attack_test(model, img_size=img_size, device=device,
                        test_dir=attack_test_dir, test_list=attack_test_list,
                        target_label=attack_target_label, log_path=attack_log_path,
                        acc_label=acc_label, normalization=normalization)

    return model, loss_history, metric_history


def test(model, test_loader, device):
    with torch.no_grad:
        model.eval()

        num_classes = len(test_loader.dataset.classes)

        test_correct_pred = 0.0

        for _, data in enumerate(test_loader):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            _, pred = torch.max(output, 1)
            test_correct_pred += (pred == label).sum()

        test_accuracy = (test_correct_pred.item() * 100) / (num_classes * 20)

    print('Test accuracy: {:.2f}%'.format(test_accuracy))
    print(test_correct_pred.item())

    # return test_accuracy


def train_val(model, train_loader, val_loader, learning_rate, device, path2weights, batch_size=32, num_epochs=10,
              attack=False, img_size=None, attack_test_dir=None, attack_test_list=None,
              attack_target_label=None, attack_log_path=None, acc_label=None,
              normalization=False
              ):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    # best_loss = float('inf')
    # best_model_wts = copy.deepcopy(model.state_dict())

    # Training
    for epoch in range(num_epochs):
        start_time = time.perf_counter()
        current_lr = get_lr(optimizer)
        print(f"Current Learning rate : {current_lr}")
        model.train()

        train_running_loss = 0.0
        train_correct_pred = 0.0
        train_num_batch = 0.0

        val_running_loss = 0.0
        val_correct_pred = 0.0
        val_num_batch = 0.0

        for index, data in enumerate(train_loader):
            train_num_batch = index
            image, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            _, pred = torch.max(output, 1)
            train_correct_pred += (pred == label).sum()

            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += float(loss.item())

        train_loss = train_running_loss / (train_num_batch + 1)
        train_accuracy = (train_correct_pred.item() / (batch_size * (train_num_batch + 1)) * 100)

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_accuracy)

        # Validation

        with torch.no_grad():
            model.eval()

            for index, data in enumerate(val_loader):
                val_num_batch = index
                image, label = data
                image = image.to(device)
                label = label.to(device)
                output = model(image)

                _, pred = torch.max(output, 1)
                val_correct_pred += (pred == label).sum()

                loss = loss_function(output, label)
                val_running_loss += float(loss.item())

            val_loss = val_running_loss / (val_num_batch + 1)
            val_accuracy = (val_correct_pred.item() / (batch_size * (val_num_batch + 1)) * 100)

        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_accuracy)

        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     torch.save(model.state_dict(), path2weights)
        #     print('Copied best model weights!')
        #
        lr_scheduler.step(val_loss)
        # if current_lr != get_lr(optimizer):
        #     print('Loading best model weights!')
        #     model.load_state_dict(best_model_wts)

        end_time = time.perf_counter()

        print('epoch {}/{}'.format(epoch + 1, num_epochs), end=" ")
        print('train loss: {:.6f}, train accuracy: {:.2f}%, val loss: {:.6f}, '
              'validation accuracy: {:.2f}%, time: {:.4f}s'.format(train_loss,
                                                                   train_accuracy,
                                                                   val_loss,
                                                                   val_accuracy,
                                                                   (end_time - start_time)))
        # print('-' * 10)

        if attack:
            attack_test(model, img_size=img_size, device=device,
                        test_dir=attack_test_dir, test_list=attack_test_list,
                        target_label=attack_target_label, log_path=attack_log_path,
                        acc_label=acc_label,
                        normalization=normalization)

        # model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history


def visualization(num_epochs, loss_hist, metric_hist):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_hist['train'], 'b-', label='train')
    # plt.plot(range(1, num_epochs + 1), loss_hist['val'], 'r--', label='val')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), metric_hist['train'], 'g-', label='train')
    # plt.plot(range(1, num_epochs + 1), metric_hist['val'], 'k--', label='val')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


def write_asr(log_path: str, asr_list: list) -> None:
    with open(log_path, "a") as file:
        for asr in asr_list:
            file.write(str(asr) + ", ")
        file.writelines("\n")
