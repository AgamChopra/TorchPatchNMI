# -*- coding: utf-8 -*-
"""
Created on Tue May 23 2023

@author: Agam Chopra
"""

import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


steps = 256
bandwidth = 2
EPSILON = 1E-10
mpl.rcParams['figure.dpi'] = 500


def print_pdf(pdf,title=None):
    for pf in pdf:
        plt.plot(pf.detach().cpu().numpy(), linewidth=0.3)
    plt.title(title)
    plt.show()


def K_gauss(input_):
    output_ = (1/(2 * torch.pi)) * torch.exp(-(input_ ** 2) / 2)

    return output_


def PDF_xis(signals, xis, h=3):  # h is the bandwidth
    x_diff = torch.stack([signals - xis[:, i:i+1]
                         for i in range(xis.shape[1])], dim=-1)

    x_ = x_diff / h
    tf = K_gauss(x_)
    p_xi = (1 / h) * torch.mean(tf, dim=1)

    return p_xi


def PDF(signals, Xs, h=3):
    pdf = PDF_xis(signals, Xs, h)

    return pdf


def get_pdf(data, steps=256, bandwidth=2):
    # [N,...] N flattened 1-d signals
    signals = torch.flatten(data, start_dim=1)

    min_val, max_val = torch.max(signals), torch.min(
        signals)  # [N] max/min values
    line_samples = torch.linspace(min_val, max_val, steps, dtype=torch.float, device=signals.device,
                                  requires_grad=False) * torch.ones((len(data), steps), dtype=torch.float, device=signals.device, requires_grad=False)

    pdf = PDF(signals, line_samples, h=bandwidth)

    return pdf


def NMI(img1, img2, bins=256, bandwidth=0.1):
    # Calculate the approx. histograms via PDF using KDE
    hist1 = get_pdf(img1, steps=bins, bandwidth=bandwidth)

    hist2 = get_pdf(img2, steps=bins, bandwidth=bandwidth)

    hist_joint = get_pdf(torch.stack((img1, img2), dim=1),
                         steps=bins, bandwidth=bandwidth)

    # Calculate the probability distributions
    p1 = hist1 / torch.unsqueeze(torch.sum(hist1, dim=1), dim=1)
    p2 = hist2 / torch.unsqueeze(torch.sum(hist2, dim=1), dim=1)
    p_joint = hist_joint / torch.unsqueeze(torch.sum(hist_joint, dim=1), dim=1)

    print_pdf(p1,'Img1 patch PDF')
    print_pdf(p2,'Img2 patch PDF')
    print_pdf(p_joint,'Img12_joint patch PDF')

    # Calculate the shannon entropy
    E1 = -torch.sum((p1 * -torch.log2(p1 + EPSILON)), dim=1)
    E2 = -torch.sum((p2 * -torch.log2(p2 + EPSILON)), dim=1)
    E_joint = -torch.sum((p_joint * -torch.log2(p_joint + EPSILON)), dim=1)

    # Calculate the mutual information
    mutual_info = E1 + E2 - E_joint
    norm_mutual_info = 2 * mutual_info / (E1 + E2)

    return norm_mutual_info, mutual_info


if __name__ == '__main__':
    path = 'B:/Users/Agam Chopra/Downloads/ringuiiii.png'
    data = Image.open(path).convert('L').resize((200, 200))
    dataA = np.array(data.getdata())
    plt.imshow(data, cmap='gray')
    plt.title('Image1')
    plt.show()

    path = 'B:/Users/Agam Chopra/Downloads/angngo.jpg'
    data = Image.open(path).convert('L').resize((200, 200))
    dataB = np.array(data.getdata())
    plt.imshow(data, cmap='gray')
    plt.title('Image2')
    plt.show()

    # [N, n-dim...] N n-dim patches of size patch_size
    moving = torch.from_numpy(dataA)
    x = moving.view(400, 10, 10).to(dtype=torch.float, device='cuda')

    target = torch.from_numpy(dataB)
    y = target.view(400, 10, 10).to(dtype=torch.float, device='cuda')

    nmi = NMI(x, y, steps, bandwidth)

    plt.plot(nmi[0].detach().cpu().numpy(), 'k-', linewidth=0.5)
    plt.plot(nmi[0].detach().cpu().numpy(), 'r_')
    plt.title('Normalized Mutual Information per Patch')
    plt.show()
