import time
import sys

from util.image import attack_image_save
from util.attack_util import *


def get_CP_loss(net_list, target_feature_list, poison_batch, s_coeff_list, tol=1e-6, bulls_eye=False):
    """
    Corresponding to one step of the outer loop (except for updating and clipping) of Algorithm 1

    :param bulls_eye: bulls-eye
    :param net_list: Classifiers list
    :param target_feature_list: Target instance's feature extracted from each classifier
    :param poison_batch: Poison batch (attack samples)
    :param s_coeff_list: Convex combination coefficient list
    :param tol: tol
    :return: total loss , Optimized Convex combination coefficient list
    """

    poison_feat_mat_list = [net(x=poison_batch(), penu=True) for net in net_list]

    # Convex combination coefficient optimization
    if bulls_eye:
        s_coeff_list = s_coeff_list

    else:
        for nn, (poison_feature, target_feature) in enumerate(zip(poison_feat_mat_list, target_feature_list)):
            s_coeff_list[nn] = least_squares_simplex(A=poison_feature.t().detach(), b=target_feature.t().detach(),
                                                     x_init=s_coeff_list[nn], tol=tol)

    total_loss = 0

    # Total_loss calculation
    for net, s_coeff, target_feature, poison_feature in zip(net_list, s_coeff_list, target_feature_list,
                                                            poison_feat_mat_list):
        # Objective function's numerator
        residual = target_feature - torch.sum(s_coeff * poison_feature, 0, keepdim=True)

        # Objective function's denominator
        target_norm_square = torch.sum(target_feature ** 2)

        # Objective function
        recon_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

        total_loss += recon_loss

    return total_loss, s_coeff_list


# Check it later
def get_CP_loss_end2end(net_list, target_feature_list, poison_batch, s_coeff_list, tol=1e-6, bulls_eye=False):
    """
    Corresponding to one step of the outer loop (except for updating and clipping) of Algorithm 1
    """
    poison_feat_mat_list = [net(x=poison_batch(), block=True) for net in net_list]

    total_loss = 0

    for nn, (net, target_feats, poison_feats) in enumerate(zip(net_list, target_feature_list, poison_feat_mat_list)):
        for n_block, (pfeat, tfeat) in enumerate(zip(poison_feats, target_feats)):
            pfeat_ = pfeat.view(pfeat.size(0), -1).t().detach()
            tfeat_ = tfeat.view(-1, 1).detach()

            if bulls_eye:
                s_coeff_list = s_coeff_list
            else:
                s_coeff_list[nn][n_block] = least_squares_simplex(A=tfeat_, b=pfeat_,
                                                                  x_init=s_coeff_list[nn][n_block], tol=tol)

            residual = tfeat - torch.sum(s_coeff_list[nn][n_block].unsqueeze(2).unsqueeze(3) * pfeat, 0, keepdim=True)
            target_norm_square = torch.sum(tfeat ** 2)
            recon_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

            total_loss += recon_loss

    return total_loss, s_coeff_list


def make_convex_polytope_poisons(subs_net_list, target_net, base_tensor_list, target,
                                 device, chk_path, opt_method='adam',
                                 lr=0.1, momentum=0.9, iterations=4000, epsilon=0.1,
                                 decay_ites=[10000, 15000], decay_ratio=0.1,
                                 mean=torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1),
                                 std=torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1),
                                 poison_idxes=[], poison_label=-1,
                                 tol=1e-6, start_ite=0, poison_init=None, end2end=False, count=None, bulls_eye=False):
    """

    :param bulls_eye:
    :param count: image count
    :param subs_net_list: Classifiers list
    :param target_net: Target model (One of the Classifiers)
    :param base_tensor_list: Base instances list
    :param target: Target instance
    :param device: Device
    :param opt_method: Optimization method
    :param lr: Learning rate
    :param momentum: Momentum for adam
    :param iterations: The number of iterations to make poison samples
    :param epsilon: Perturbation range
    :param decay_ites: Iterations condition to decay the optimizer learning rate
    :param decay_ratio: Optimizer learning rate decay ratio
    :param mean: Predefined mean
    :param std: Predefined std
    :param chk_path:
    :param poison_idxes:
    :param poison_label: poison label
    :param tol:
    :param start_ite: 0
    :param poison_init: Initial status of poison batch (samples)
    :param end2end: end2end
    :return: get_poison_tuples & total loss
    """

    target_net.eval()

    # Poison samples 생성을 위한 poison_batch 생성
    poison_batch = PoisonBatch(poison_init).to(device)

    # Select the optimization method
    opt_method = opt_method.lower()
    if opt_method == 'sgd':
        optimizer = torch.optim.SGD(poison_batch.parameters(), lr=lr, momentum=momentum)

    elif opt_method == 'adam':
        optimizer = torch.optim.Adam(poison_batch.parameters(), lr=lr, betas=(momentum, 0.999))

    # Variable to GPU
    target = target.to(device)
    std, mean = std.to(device), mean.to(device)

    base_tensor_batch = torch.stack(base_tensor_list, 0)
    base_tensor_batch = base_tensor_batch.to(device)
    base_range01_batch = base_tensor_batch * std + mean

    # Because we have turned on DP for the substitute networks,
    # the target image's feature becomes random.
    # We can try enforcing the convex polytope in one of the multiple realizations of the feature,
    # but empirically one realization is enough.
    target_feat_list = []

    # Coefficients for the convex combination.
    # Initializing from the coefficients of last step gives faster convergence.
    s_init_coeff_list = []

    n_poisons = len(base_tensor_list)

    # loss_compare_list = []
    #
    # for poison in range(n_poisons):
    #     loss_compare_list.append([])

    # Initialize the Convex combination coefficient
    for n, net in enumerate(subs_net_list):
        net.eval()

        if end2end:
            block_feats = [feat.detach() for feat in net(x=target, block=True)]
            target_feat_list.append(block_feats)
            s_coeff = [torch.ones(n_poisons, 1).to(device) / n_poisons for _ in range(len(block_feats))]

        else:
            target_feat_list.append(net(x=target, penu=True).detach())
            s_coeff = torch.ones(n_poisons, 1).to(device) / n_poisons

        s_init_coeff_list.append(s_coeff)

    # Keep this for evaluation.
    if end2end:
        target_feat_in_target = [feat.detach() for feat in target_net(x=target, block=True)]
        target_init_coeff = [[torch.ones(n_poisons, 1).to(device) / n_poisons
                              for _ in range(len(target_feat_in_target))]]
    else:
        target_feat_in_target = target_net(x=target, penu=True).detach()
        target_init_coeff = [torch.ones(n_poisons, 1).to(device) / n_poisons]

    cp_loss_func = get_CP_loss_end2end if end2end else get_CP_loss

    for ite in range(start_ite, iterations):
        if ite in decay_ites:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_ratio
            print("%s Iteration %d, Adjusted lr to %.2e" % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, lr))

        # Poison sample's gradient initializing
        poison_batch.zero_grad()

        # choosing a realization of the target
        # Total loss = Objective function's result , s_coeff_list = Optimized Convex combination efficient
        total_loss, s_coeff_list = cp_loss_func(subs_net_list, target_feat_list, poison_batch, s_init_coeff_list,
                                                tol=tol, bulls_eye=bulls_eye)

        total_loss.backward()
        optimizer.step()

        # clip the perturbations into the range

        # Pixel level에서 (Poison samples - Base instances)*std 의 값이 (-0.1 ~ 0.1) 사이가 되도록 perturb_range 설정

        # Original
        # perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch) * std, -epsilon, epsilon)
        # Test
        perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch), -epsilon, epsilon)

        # Pixel level에서 Normalized base instances + perturbation 범위는 (0 ~ 1)로 고정
        # perturbed_range01 = torch.clamp(base_range01_batch.data + perturb_range01.data, 0, 1)
        # Test
        perturbed_range01 = torch.clamp(base_tensor_batch.data + perturb_range01.data, 0, 1).detach()

        # Original
        # poison_batch.poison.data = (perturbed_range01 - mean) / std
        # Test
        poison_batch.poison.data = perturbed_range01

        if ite % 500 == 0 or ite == iterations - 1:
            target_loss, target_coeff = cp_loss_func([target_net], [target_feat_in_target], poison_batch,
                                                     target_init_coeff, tol=tol)

            # compute the difference in target
            print(" %s Iteration %d \t Training Loss: %.3e \t Loss in Target Net: %.3e\t  " % (
                time.strftime("%Y-%m-%d %H:%M:%S"), ite, total_loss.item(), target_loss.item()))
            print(target_coeff)
            sys.stdout.flush()

            # save the checkpoints
            # poison_tuple_list = get_poison_tuples(poison_batch, poison_label)
            # torch.save({'poison': poison_tuple_list, 'idx': poison_idxes},
            #            os.path.join(chk_path, "poison_%05d.pth" % ite))

        # if ite % 2000 == 0 or ite == iterations - 1:
        if ite == iterations - 1:

            # Poison batch에서 instances 분리
            poison_instances = poison_batch.poison.data.detach()

            # Poison instances에서 각 instance 추출하여 저장
            for idx in range(len(poison_instances)):
                instance = poison_instances[idx, :]
                attack_image_save(instance, chk_path, f"test_cp_attack_{ite}_{(count * 5) + idx}.png")

    return get_poison_tuples(poison_batch, poison_label), total_loss.item()
