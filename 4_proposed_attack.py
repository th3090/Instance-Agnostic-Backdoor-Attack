from torchsummary import summary

from attack.convex_polytope import make_convex_polytope_poisons
from util.attack_util import *
from attack.proposed_method import make_poisons
from attack.poison_frogs import poison_frogs

from model.alexnet import alexnet
from model.vgg import vgg16
from util.image import attack_image_save
from util.training import set_data_loader, train_val


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = (224, 224)

    save_path = './new_dataset/Training_105/Z_Taehoon4'
    log_path = './results/FC_Penu_Vgg_single_generator.txt'

    # base = './new_dataset/Attack_Source/Taehoon_benign_10'
    base = './new_dataset/Attack_Source/Taehoon_benign'
    target = './new_dataset/Attack_Source/Single_target_trigger'

    # Generator setting
    pretrained = True
    end2end = False
    check_point = False
    normalization = False

    penu = True
    fc1_extractor = False
    block = False

    attack_modes = 'proposed'
    # attack_modes = 'convex_polytope'

    target_net = vgg16(pretrained=pretrained, check_point=check_point)
    # target_net = alexnet(pretrained=pretrained, check_point=check_point)

    target_net = target_net.to(device)
    summary(target_net, (3, 224, 224), device=device.type)

    target_net.eval()

    if attack_modes == 'proposed':

        sub_net_list = [target_net]

        base_tensor_list = set_base_tensor_list(base_path=base, img_size=img_size,
                                                normalization=normalization)
        img_count = 0
        idx = 0

        for i in range(len(base_tensor_list)):
            target_tensor_list = set_target_tensor_list(target_path=target, img_size=img_size,
                                                        random_sample=True, number=5,
                                                        normalization=normalization)

            _, total_loss, attack_instance = make_poisons(sub_net_list, target_net, base_tensor_list[i],
                                                          target_tensor_list,
                                                          device, save_path, opt_method='adam', idx=idx,
                                                          img_count=img_count,
                                                          learning_rate=0.001, momentum=0.9, iterations=2000,
                                                          epsilon=8 / 255,
                                                          decay_ites=[10000, 15000], decay_ratio=0.1,
                                                          poison_label=-1, tol=1e-6,
                                                          start_ite=0, end2end=end2end,
                                                          penu=penu, fc1_extractor=fc1_extractor, block=block)

            img_count += 1
            with open(log_path, 'a') as file:
                file.writelines("\n" + str(total_loss))
            print(total_loss)
            print("-------------------------------------------")

            # print("Base instance - Poisoned instance: {}".format(
            #     feature_loss_compare(target_net, base_tensor_list[0], attack_instance)))

    elif attack_modes == 'convex_polytope':
        sub_net_list = [target_net]

        base_img_list = []
        total_base_tensor_list = []

        for i in os.listdir(base):
            base_img_list.append(i)

        for img in range(len(base_img_list)):
            base_image = img_read(os.path.join(base + "/" + base_img_list[img]))

            # image를 tensor 형태로 변환 후 base_list에 저장
            base_instance = img_to_tensor(base_image, img_size)
            total_base_tensor_list.append(base_instance)

        for i in range(4):
            target_tensor = set_target_tensor_list(target_path=target, img_size=img_size,
                                                   random_sample=True, number=1)

            base_tensor_list = total_base_tensor_list[(i * 5):(i + 1) * 5]

            poison_init = base_tensor_list

            _, total_loss = make_convex_polytope_poisons(sub_net_list, target_net, base_tensor_list,
                                                         target_tensor,
                                                         device, save_path, opt_method='adam',
                                                         lr=0.001, momentum=0.9, iterations=2000,
                                                         epsilon=8 / 255,
                                                         decay_ites=[10000, 15000], decay_ratio=0.1,
                                                         mean=torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1,
                                                                                                             3,
                                                                                                             1,
                                                                                                             1),
                                                         std=torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1,
                                                                                                            3,
                                                                                                            1,
                                                                                                            1),
                                                         poison_idxes=[], poison_label=-1,
                                                         tol=1e-6, start_ite=0, poison_init=poison_init,
                                                         end2end=False, count=i, bulls_eye=True)

        print(total_loss)

    # Training
    remove_list = ["Z_Taehoon1", "Z_Taehoon2", "Z_Taehoon3"]

    num_epochs = 80
    img_size = (224, 224)
    workers = 0
    batch_size = 32
    learning_rate = 0.001

    # train_dir = './new_dataset/Training_55'
    # test_dir = './new_dataset/Test_55'

    train_dir = './new_dataset/Training_105'
    test_dir = './new_dataset/Test_105'

    attack_test_dir = './new_dataset/Attack_Test'
    attack_test_list = ["Gyumin_trigger1", "Gyumin_trigger1_aug", "Hyeju_trigger1", "Jinmyeong_trigger1",
                        "Sunjin_trigger1", "Taehoon_trigger1"]

    # attack_test_label = [20, 20, 20, 21, 22, 23, 24]
    attack_test_label = [100, 100, 101, 102, 103, 104]
    attack_target_label = 104

    # attack_test_label = [50, 50, 51, 52, 53, 54]
    # attack_target_label = 54

    attack_log_path = './results/attack_log3.txt'
    path2weights = './result/trained_weight.pth'

    # Trainer setting
    pretrained = True
    classifier_target_layer = 6

    train_loader = set_data_loader(train_dir, img_size=img_size, workers=workers,
                                   batch_size=batch_size, train=True,
                                   remove_list=remove_list)
    test_loader = set_data_loader(test_dir, img_size=img_size, workers=workers,
                                  batch_size=batch_size, train=False,
                                  remove_list=remove_list)

    num_classes = len(train_loader.dataset.classes)

    model = vgg16(pretrained=pretrained)
    # model = alexnet(pretrained=pretrained)
    model.pretrain_layer(num_classes=num_classes, classifier_target_layer=classifier_target_layer)
    model = model.to(device)
    summary(model, (3, 224, 224), device=device.type)

    model, loss_hist, metric_hist = train_val(model, train_loader, test_loader, learning_rate=learning_rate,
                                              device=device,
                                              path2weights=path2weights,
                                              batch_size=batch_size,
                                              num_epochs=num_epochs,
                                              attack=True, img_size=img_size,
                                              attack_test_dir=attack_test_dir,
                                              attack_test_list=attack_test_list,
                                              attack_target_label=attack_target_label,
                                              attack_log_path=attack_log_path,
                                              acc_label=attack_test_label)

    SAVE_PATH = ('./check_point/attack_1_target_layer{}.pth'.format(classifier_target_layer))
    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    main()
