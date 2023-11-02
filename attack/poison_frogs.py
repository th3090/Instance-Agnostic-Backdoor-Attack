import torch
import torch.nn as nn


def poison_frogs(extractor, attack_instance, base_instance, target_instance, end2end=False,
                 epsilon=8 / 255, alpha=0.05 / 255):
    attack_instance.requires_grad = True
    extractor.eval()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    mean = torch.Tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    std = torch.Tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)

    mean = mean.to(device)
    std = std.to(device)

    if end2end:
        # target_instance_feature = extractor(((target_instance.view(1, *target_instance.shape)-mean)/std), block=True)  # Normalization
        target_instance_feature = extractor(target_instance.view(1, *target_instance.shape), block=True)

        # attack_instance_feature = extractor(((attack_instance.view(1, *attack_instance.shape)-mean)/std), block=True) # Normalization
        attack_instance_feature = extractor(attack_instance.view(1, *attack_instance.shape), block=True)

        feature_loss = nn.MSELoss()(attack_instance_feature[0][0], target_instance_feature[0][0])

        for n, (target_feats, poison_feats) in enumerate(zip(target_instance_feature, attack_instance_feature)):
            target_feats = target_feats[0].detach()
            target_feats.requires_grad = False

            poison_feats = poison_feats[0]

            n_layer_loss = nn.MSELoss()(poison_feats, target_feats)
            feature_loss += n_layer_loss

        image_loss = nn.MSELoss()(attack_instance, base_instance)
        loss = feature_loss + image_loss / 1e2
        loss.backward()

        signed_gradient = attack_instance.grad.sign()

        attack_instance = attack_instance - alpha * signed_gradient
        eta = torch.clamp(attack_instance - base_instance, -epsilon, epsilon)
        attack_instance = torch.clamp(base_instance + eta, 0, 1).detach()

        return attack_instance, loss.item()

    else:
        target_instance_feature = extractor(target_instance.view(1, *target_instance.shape), penu=True)[0].detach()
        # target_instance_feature = extractor(target_instance.view(1, *target_instance.shape))[0].detach()
        # target_instance_feature = extractor(((target_instance.view(1, *target_instance.shape) - mean)/std))[0].detach()
        target_instance_feature.requires_grad = False  # .detach() 사용할 경우 gradient 전파 안되는 텐서를 생성

        attack_instance_feature = extractor(attack_instance.view(1, *attack_instance.shape), penu=True)[0]
        # attack_instance_feature = extractor(attack_instance.view(1, *attack_instance.shape))[0]
        # attack_instance_feature = extractor(((attack_instance.view(1, *attack_instance.shape) - mean)/std))[0]

        # Forward Step:
        feature_loss = nn.MSELoss()(attack_instance_feature, target_instance_feature)
        image_loss = nn.MSELoss()(attack_instance, base_instance)
        loss = feature_loss + image_loss / 1e2
        loss.backward()

        signed_gradient = attack_instance.grad.sign()

        attack_instance = attack_instance - alpha * signed_gradient
        eta = torch.clamp(attack_instance - base_instance, -epsilon, epsilon)
        attack_instance = torch.clamp(base_instance + eta, 0, 1).detach()

        return attack_instance, loss.item()
