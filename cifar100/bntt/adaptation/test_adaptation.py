import torch
from .aug_mix32 import aug32


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


def adapt(model, optimizer, image, niter=1, batch_size=32, method='space'):
    for _ in range(niter):
        images = [aug32(image) for _ in range(batch_size)]
        images = torch.stack(images).cuda()

        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        outputs, hiddens = model(images)
        loss = einsum(hiddens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
