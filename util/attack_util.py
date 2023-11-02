# 공격할 때 공통적으로 사용하는 함수들 여기 집어 넣을 것
import os
import random
from typing import List

import numpy as np
import torch
from torch import nn

from util.image import img_to_tensor, img_read


class PoisonBatch(nn.Module):
    """
    Implementing this to work with PyTorch optimizers.
    """

    def __init__(self, base_list):
        super(PoisonBatch, self).__init__()
        base_batch = torch.stack(base_list, 0)
        self.poison = torch.nn.Parameter(base_batch.clone())

    def forward(self):
        return self.poison


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels

    :param poison_batch: poison_batch
    :param poison_label: poison_label
    :return: tuple
    """

    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in
                    range(poison_batch.poison.size(0))]

    return poison_tuple


def proj_onto_simplex(coeffs, psum=1.0):
    """
    For original convex polytope attack

    Code stolen from https://github.com/hsnamkoong/robustopt/blob/master/src/simple_projections.py
    Project onto probability simplex by default.
    """
    v_np = coeffs.view(-1).detach().cpu().numpy()
    n_features = v_np.shape[0]
    v_sorted = np.sort(v_np)[::-1]
    cssv = np.cumsum(v_sorted) - psum
    ind = np.arange(n_features) + 1
    cond = v_sorted - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w_ = np.maximum(v_np - theta, 0)
    return torch.Tensor(w_.reshape(coeffs.size())).to(coeffs.device)


def least_squares_simplex(A, b, x_init, tol=1e-6, verbose=False, device='cuda'):
    """
    For original convex polytope attack
    The inner loop of Algorithm 1
    :param A: Poison batch's feature which is extracted by classifier
    :param b: Target instance's feature which is extracted by classifier
    :param x_init: Initial convex combination coefficient
    :param tol: Stopping condition
    :param verbose: verbose
    :param device: GPU device
    :return: Optimized convex combination coefficient
    """

    # m = The number of features , n = The number of poison batch's data
    m, n = A.size()

    # b.size = (The number of features , 1)
    assert b.size()[0] == A.size()[0], 'Matrix and vector do not have compatible dimensions'

    # Initialize the optimization variables
    if x_init is None:
        x = torch.zeros(n, 1).to(device)
    else:
        x = x_init

    # Define the objective function and its gradient
    feature_loss = lambda x: torch.norm(A.mm(x) - b).item()  # 4096x5 * 5x1 = 4096x1 - 4096x1 --> norm value

    # change into a faster version when A is a tall matrix
    AtA = A.t().mm(A)  # nx4096 * 4096xn = nxn
    Atb = A.t().mm(b)  # nx4096 * 4096x1 = nx1
    grad_f = lambda x: AtA.mm(x) - Atb  # nxn * nx1 = nx1 - nx1
    # grad_f = lambda x: A.t().mm(A.mm(x)-b)

    # Estimate the spectral radius of the Matrix A'A
    y = torch.normal(0, torch.ones(n, 1)).to(device)
    lipschitz = torch.norm(A.t().mm(A.mm(y))) / torch.norm(y)  # nx4096 * (4096xn * nx1) = nx1

    # The step size for the problem should be 2/lipschitz.  Our estimator might not be correct, it could be too small.
    # In this case our learning rate will be too big, and so we need to have a backtracking line search to make sure things converge.
    t = 2 / lipschitz

    # Main iteration
    for ite in range(10000):
        x_hat = x - t * grad_f(x)  # Forward step:  Gradient decent on the objective term
        if feature_loss(x_hat) > feature_loss(
                x):  # Check whether the learning rate is small enough to decrease objective
            t = t / 2
        else:
            x_new = proj_onto_simplex(x_hat)  # Backward step: Project onto prob simplex
            stopping_condition = torch.norm(x - x_new) / max(torch.norm(x), 1e-8)

            if not verbose:
                pass
            else:
                print('iter %d: error = %0.4e' % (ite, stopping_condition))

            if stopping_condition < tol:  # check stopping conditions
                break
            x = x_new

    return x


def filename_from_dir(base_path) -> List:
    filename_list = []
    for i in os.listdir(base_path):
        filename_list.append(i)
    return filename_list


def set_base_tensor_list(base_path, img_size=(224, 224), normalization=False):
    img_list = filename_from_dir(base_path)
    tensor_list = []

    for img in range(len(img_list)):
        image = img_read(os.path.join(base_path + "/" + img_list[img]))
        base_instance = img_to_tensor(image, img_size, normalization=normalization)
        base_tensor = [base_instance]
        tensor_list.append(base_tensor)

    return tensor_list


def set_target_tensor_list(target_path, img_size=(224, 224),
                           random_sample=False, number=None, normalization=False):
    target_tensor_list = []
    target_img_list = filename_from_dir(target_path)

    for img in range(len(target_img_list)):
        target_image = img_read(os.path.join(target_path + "/" + target_img_list[img]))
        # image를 tensor 형태로 변환 후 base_list에 저장
        target_instance = img_to_tensor(target_image, img_size, normalization=normalization)
        target_tensor_list.append(target_instance)

        # 랜덤 이미지 n개 선정
    if random_sample:
        random_target_list = random.sample(target_tensor_list, number)
        # print(len(random_target_list))
        target_tensor_list = torch.stack(random_target_list, dim=0)

    if not random_sample:
        target_tensor_list = torch.stack(target_tensor_list, dim=0)

    return target_tensor_list


def feature_loss_compare(extractor, instance_a, instance_b):
    instance_a_feature = extractor(instance_a.view(1, *instance_a.shape), penu=True)[0]
    instance_b_feature = extractor(instance_b.view(1, *instance_b.shape), penu=True)[0]

    dif = instance_a_feature - instance_b_feature
    loss = torch.sum(torch.mul(dif, dif))

    return loss.item()