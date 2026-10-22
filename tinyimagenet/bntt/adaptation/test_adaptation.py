import torch
from .aug_mix64 import aug64
import numpy as np
import sys


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cross_entropy(hs):
    hs = hs.view(hs.size(0), -1)
    hs = torch.nn.functional.softmax(hs, dim=-1)
    ce = []
    for i in range(hs.size(0)):
        for j in range(i + 1, hs.size(0)):
            ce.append(-torch.sum(hs[i] * torch.log(hs[j])))
    return torch.mean(torch.stack(ce))


def einsum(hs, scale=11):
    batch_size, channels, _, _ = hs.size()
    hs = hs.view(batch_size, channels, -1)
    normalized_hs = torch.nn.functional.softmax(hs, dim=-1)
    similarity = torch.einsum('bck,dck->cbd', normalized_hs, normalized_hs)
    mask = torch.ones_like(similarity)
    mask[0].fill_diagonal_(0)
    similarity = similarity * mask
    dissimilarity = 1. - similarity
    return scale * dissimilarity.sum() / ((batch_size * (batch_size - 1)) * channels)


def einsum_full(hs, scale=11):
    batch_size, _, _, _ = hs.size()
    hs = hs.view(batch_size, -1)
    normalized_hs = torch.nn.functional.softmax(hs, dim=-1)
    similarity = torch.einsum('bi,di->bd', normalized_hs, normalized_hs)
    mask = torch.ones_like(similarity)
    mask.fill_diagonal_(0)
    similarity = similarity * mask
    dissimilarity = 1. - similarity
    return scale * dissimilarity.sum() / (batch_size * (batch_size - 1))


def compute_batch_stats(model, aug_images):
    model.eval()
    batch_stats = {}
    with torch.no_grad():
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.track_running_stats = False

                def hook_fn(module, input, output):
                    batch_stats[name] = {
                        'mean': input[0].mean(dim=(0, 2, 3) if isinstance(module, torch.nn.BatchNorm2d) else 0),
                        'var': input[0].var(dim=(0, 2, 3) if isinstance(module, torch.nn.BatchNorm2d) else 0, unbiased=False)
                    }
                hooks.append(module.register_forward_hook(hook_fn))
        _, _ = model(aug_images)
        for hook in hooks:
            hook.remove()
    return batch_stats


def adapt(model, optimizer, image, niter=1, batch_size=32, method='ls'):
    for _ in range(niter):
        images = [aug64(image) for _ in range(batch_size)]
        images = torch.stack(images).cuda()

        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        outputs, hiddens = model(images)
        loss = einsum_full(hiddens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
