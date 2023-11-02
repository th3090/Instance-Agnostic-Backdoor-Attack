import time
import sys

from util.attack_util import *
from util.image import attack_image_save


def get_single_layer_loss(net_list, target_feature_list, poison_batch, cc_coeff, tol=1e-6, bulls_eye=True,
                          penu=True, fc1_extractor=False, block=False):
    """
    Modify the "get_CP_loss" to implement N:M mapping CP Attack
    한 개의 layer에서 feature transfer를 진행하는 경우

    :param fc1_extractor:
    :param block:
    :param penu:
    :param tol:
    :param net_list: Classifiers list
    :param bulls_eye: bulls_eye option
    :param target_feature_list: Target instance's feature extracted from target model. This values will not change.
    :param poison_batch: Poison instance (attack sample)
    :param cc_coeff: Convex combination coefficient list
    :return: Total loss, Optimized Convex combination coefficient list
    """

    # Poison instance feature 추출
    # poison_instance_feature = [net(x=poison_batch()) for net in net_list]
    poison_instance_feature = [net(x=poison_batch(), penu=penu, fc1_extractor=fc1_extractor, block=block) for net in net_list]

    if not bulls_eye:
        raise NotImplementedError("Proposed method supports only bulls_eye polytope!")

    cc_coeffs = [cc_coeff]

    total_loss = 0

    # Total_loss calculation
    for net, cc_coeff, target_feature, poison_feature in zip(net_list, cc_coeffs, target_feature_list,
                                                             poison_instance_feature):
        # Objective function's numerator
        residual = poison_feature - torch.sum(cc_coeff * target_feature, 0, keepdim=True)

        # Objective function's denominator
        poison_norm_square = torch.sum(poison_feature ** 2)

        # Objective function
        recon_loss = 0.5 * torch.sum(residual ** 2) / poison_norm_square

        total_loss += recon_loss

    return total_loss, cc_coeff


def get_multi_layer_loss(net_list, target_feature_list, poison_batch, s_coeff_list, tol=1e-6, bulls_eye=True):
    """
    Corresponding to one step of the outer loop (except for updating and clipping) of Algorithm 1
    """
    poison_instance_feature = [net(x=poison_batch(), block=True) for net in net_list]

    if not bulls_eye:
        raise NotImplementedError("Proposed method supports only bulls_eye polytope!")

    total_loss = 0

    for nn, (net, target_feature, poison_feature) in enumerate(
            zip(net_list, target_feature_list, poison_instance_feature)):
        total_loss_tmp = 0
        for n_block, (tfeat, pfeat) in enumerate(zip(target_feature, poison_feature)):
            s_coeff_list[nn][n_block] = least_squares_simplex(A=tfeat.view(tfeat.size(0), -1).t().detach(),
                                                              b=pfeat.view(-1, 1).detach(),
                                                              x_init=s_coeff_list[nn][n_block], tol=tol)

            residual = tfeat - torch.sum(s_coeff_list[nn][n_block].unsqueeze(2).unsqueeze(3) * pfeat, 0, keepdim=True)
            target_norm_square = torch.sum(tfeat ** 2)
            recon_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

            total_loss_tmp += recon_loss
        total_loss += total_loss_tmp / len(poison_feature)

    total_loss = total_loss / len(net_list)

    return total_loss, s_coeff_list


def loss_from_center(subs_net_list, target_feat_list, poison_batch, device, end2end=False, test=False,
                     penu=True, fc1_extractor=False, block=False):

    if test:
        optimizer_loss = 0

        for net, center_feats in zip(subs_net_list, target_feat_list):
            # poisons_feats = net(x=poison_batch())
            poisons_feats = net(x=poison_batch(), penu=penu, fc1_extractor=fc1_extractor, block=block)

            feature_loss = nn.MSELoss()(poisons_feats, center_feats[0])

            target_feature = torch.mean(center_feats, dim=0, keepdim=True)
            target_feature_norm = torch.norm(target_feature, dim=1)

            diff = torch.mean(center_feats, dim=0) - poisons_feats
            diff_norm = torch.norm(diff, dim=1) / target_feature_norm

            optimizer_loss += torch.mean(diff_norm) + 5*feature_loss

        optimizer_loss = optimizer_loss / len(subs_net_list)
        return optimizer_loss

    if end2end:
        optimizer_loss = 0

        for net, center_feats in zip(subs_net_list, target_feat_list):
            poisons_feats = net(x=poison_batch(), block=True)

            opt_loss = 0

            for pfeat, cfeat in zip(poisons_feats, center_feats):
                target_feature = torch.mean(cfeat, dim=0)
                target_feature_norm = torch.norm(target_feature, dim=0)

                diff = target_feature - pfeat
                diff_norm = torch.norm(diff, dim=1) / target_feature_norm

                opt_loss += torch.mean(diff_norm)

            optimizer_loss += opt_loss / len(center_feats)

        optimizer_loss = optimizer_loss / len(subs_net_list)

    else:
        optimizer_loss = 0

        for net, center_feats in zip(subs_net_list, target_feat_list):
            # poisons_feats = net(x=poison_batch())
            poisons_feats = net(x=poison_batch(), penu=penu, fc1_extractor=fc1_extractor, block=block)

            target_feature = torch.mean(center_feats, dim=0, keepdim=True)
            target_feature_norm = torch.norm(target_feature, dim=1)

            diff = torch.mean(center_feats, dim=0) - poisons_feats
            diff_norm = torch.norm(diff, dim=1) / target_feature_norm

            optimizer_loss += torch.mean(diff_norm)

        optimizer_loss = optimizer_loss / len(subs_net_list)

    return optimizer_loss


def make_poisons(subs_net_list, target_net, base_tensor, target_tensor_list,
                 device, chk_path, idx, img_count, opt_method='adam',
                 learning_rate=0.001, momentum=0.9, iterations=2000, epsilon=8/255,
                 decay_ites=None, decay_ratio=0.1,
                 poison_label=-1, tol=1e-6, start_ite=0, end2end=False, test=False,
                 penu=False, fc1_extractor=False, block=False):

    # mean = torch.Tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    # std = torch.Tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    # mean = mean.to(device)
    # std = std.to(device)

    # Set target net (Up to feature transfer layer)
    if decay_ites is None:
        decay_ites = [10000, 15000]

    # Make Poison batch for poison samples from base_tensor -> (1 x 3 x 224 x 224)
    poison_batch = PoisonBatch(base_tensor).to(device)

    # Select the optimization method
    opt_method = opt_method.lower()
    if opt_method == 'sgd':
        optimizer = torch.optim.SGD(poison_batch.parameters(), lr=learning_rate, momentum=momentum)

    elif opt_method == 'adam':
        optimizer = torch.optim.Adam(poison_batch.parameters(), lr=learning_rate, betas=(momentum, 0.999))

    else:
        raise NotImplementedError("Proposed method supports only bulls_eye polytope!")

    # Variable to GPU
    n_targets = len(target_tensor_list)

    # target.shape = (n_targets x 3 x 224 x 224)
    target = target_tensor_list.to(device)

    # base.shape = (1 x 3 x 224 x 224)
    base = torch.stack(base_tensor, 0)
    base = base.to(device)

    # Coefficients for the convex combination.
    # Initializing from the coefficients of last step gives faster convergence.
    cc_init_coeff_list = []
    target_feat_list = []

    for n, net in enumerate(subs_net_list):
        net.eval()

        if end2end:
            block_feats = [feat.detach() for feat in net(x=target, block=True)]
            target_feat_list.append(block_feats)
            cc_coeff = [torch.ones(n_targets, 1).to(device) / n_targets for _ in range(len(block_feats))]

        else:
            target_feat_list.append(net(x=target, penu=penu, fc1_extractor=fc1_extractor, block=block).detach())
            # target_feat_list.append(net(x=target).detach())
            cc_coeff = torch.ones(n_targets, 1).to(device) / n_targets

        cc_init_coeff_list.append(cc_coeff)

    # Keep this for evaluation.
    if end2end:
        target_feat_in_target = [feat.detach() for feat in target_net(x=target, block=True)]
        target_init_coeff = [[torch.ones(n_targets, 1).to(device) / n_targets
                              for _ in range(len(target_feat_in_target))]]
    else:
        target_feat_in_target = target_net(x=target, penu=penu, fc1_extractor=fc1_extractor, block=block).detach()
        # target_feat_in_target = target_net(x=target).detach()
        target_init_coeff = torch.ones(n_targets, 1).to(device) / n_targets

    cp_loss_func = get_multi_layer_loss if end2end else get_single_layer_loss

    for ite in range(start_ite, iterations):
        if ite in decay_ites:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_ratio
            print("%s Iteration %d, Adjusted lr to %.2e" % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, learning_rate))

        # Poison sample's gradient initializing
        poison_batch.zero_grad()
        total_loss = loss_from_center(subs_net_list, target_feat_list, poison_batch, device=device, end2end=end2end, test=test,
                                      penu=penu, fc1_extractor=fc1_extractor, block=block)
        # total_loss.requires_grad_(True)
        total_loss.backward()
        optimizer.step()

        perturbation = torch.clamp((poison_batch.poison.data - base), -epsilon, epsilon)
        attack_instance = torch.clamp(base.data + perturbation.data, 0, 1).detach()

        poison_batch.poison.data = attack_instance

        if ite % 500 == 0 or ite == iterations - 1:
            target_loss, target_coeff = cp_loss_func([target_net], [target_feat_in_target], poison_batch,
                                                     target_init_coeff, tol=tol, penu=penu, fc1_extractor=fc1_extractor, block=block)

            # compute the difference in target
            print(" %s Iteration %d \t Training Loss: %.3e \t Loss in Target Net: %.3e\t  " % (
                time.strftime("%Y-%m-%d %H:%M:%S"), ite, total_loss.item(), target_loss.item()))
            sys.stdout.flush()

            # save the checkpoints
            # poison_tuple_list = get_poison_tuples(poison_batch, poison_label)
            # torch.save({'poison': poison_tuple_list, 'idx': poison_idxes},
            #            os.path.join(chk_path, "poison_%05d.pth" % ite))

        if ite == iterations - 1:

            # Poison batch에서 instances 분리
            poison_instances = poison_batch.poison.data.detach()

            # Poison instances에서 각 instance 추출하여 저장
            instance = poison_instances[0, :]
            attack_image_save(instance, chk_path, f"proposed_method_{idx}_{img_count}.png")

    return get_poison_tuples(poison_batch, poison_label), total_loss.item(), instance