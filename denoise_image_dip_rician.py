import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
# from utils import TVLoss
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
# from utils import add_gaussian_noise, add_rician_noise, set_random_seed
from _utils import add_gaussian_noise, add_rician_noise, crop_image

from torch import Tensor, optim, from_numpy

import sys
import pandas as pd

def rician_distribution(u, f, sigma):
    arg = (u * f) / (sigma ** 2)
    return torch.special.i1(arg) / torch.special.i0(arg)

def dip_rician(f, sigma, lda=None, lr=1e-3, outter_nits=100, dev = 'cuda', org=None, plot=None, NN=None, loss_fn='mse'):
    from torch.special import i0, i1
    from torch import log
    from denoising_net import Net
    from torch import Tensor, optim, from_numpy

    def rician_fidelity(noisy_img: Tensor, net_img: Tensor, std):
        return ((net_img.square() / (2 * std ** 2)) - log(i0((noisy_img.mul(net_img)) / std ** 2))).mean()

    def rician_norm_fidelity(f: Tensor, net_img: Tensor, **kwargs):
        return (net_img*(i0(net_img*f/(sigma**2))/i1(net_img*f/(sigma**2))) - f).square().mean() # I0 < I1, así que I0/I1 parece que funciona

    def mse_fidelity(noisy_img: Tensor, net_img: Tensor, **kwargs):
        return (noisy_img - net_img).square().mean()

    if len(f.shape) == 2:
        f = np.expand_dims(f, 0)
        f = torch.from_numpy(f)[None, ...].to(dev).permute(0, 3, 1, 2)
        b, c, w, h = f.size()
    elif len(f.shape) == 3 and f.shape[2] == 3:
        f = torch.from_numpy(f)[None, ...].to(dev).permute(0, 3, 1, 2)
        b, c, w, h = f.size()
    else:
        raise Exception("Unknown channel configuration")

    if NN is None:
        NN = Net()

    # if lda is not None:
    #     reg = TVLoss()

    if loss_fn == 'mse':
        loss_fn = mse_fidelity
    elif loss_fn == 'rician':
        loss_fn = rician_fidelity
    # elif loss_fn == 'rician_norm':
    else:
        loss_fn = rician_norm_fidelity

    seed_cpu = from_numpy(np.random.uniform(low=0, high=0.1, size=(c, w, h)).astype('float32'))[None, :]

    opt = torch.optim.Adam(lr=lr, params=NN.parameters())

    psnr_list = []
    loss_list = []
    best_u = f.clone()
    psnr_value_prev = 0
    for it in range(outter_nits):
        z = add_gaussian_noise(seed_cpu, 0, 1/20).to(dev)

        # rician loss
        u = NN(z)
        # if lda is None:
        loss = loss_fn(f, u, std=sigma)
        # else:
        #     loss = loss_fn(f, u, std=sigma) + lda*reg(u)


        opt.zero_grad()
        loss.backward()
        opt.step()

        # log
        psnr_value = psnr(np.float32(org), u[0, ...].permute(1, 2, 0).detach().cpu().numpy())
        if psnr_value_prev < psnr_value:
            best_u = u.detach().cpu().permute(0, 2, 3, 1)[0, ...]
            psnr_value_prev = psnr_value
        loss_list.append(loss.detach().cpu().numpy())
        psnr_list.append(psnr_value)

        print('Progress: {}/{} | Loss: {:.3f} | PSNR: {:.3f} | Best PSNR: {:.3f}'.format(it+1, outter_nits, loss_list[-1], psnr_list[-1], psnr_value_prev))

        if plot is not None:
            if it % plot == 0:
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(org, cmap='gray')
                axs[0, 0].set_title('org')
                axs[0, 1].imshow(f.permute(0, 2, 3, 1).cpu()[0, ...], cmap='gray')
                axs[0, 1].set_title('f')
                axs[1, 0].imshow(u.detach().cpu().permute(0, 2, 3, 1)[0, ...], cmap='gray')
                axs[1, 0].set_title('uk')
                axs[1, 1].imshow(torch.abs(f.permute(0, 2, 3, 1) - u.permute(0, 2, 3, 1).detach()).cpu()[0, ...],
                                 cmap='gray')
                axs[1, 1].set_title('dif')
                plt.show()

    outputs = {'psnr': psnr_list,
               'loss': loss_list,
               'f': f.permute(0, 2, 3, 1).cpu()[0, ...],
               'u': u.detach().cpu().permute(0, 2, 3, 1)[0, ...],
               'best_u': best_u
               }
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(org, cmap='gray')
    axs[0, 0].set_title('org')
    axs[0, 1].imshow(f.permute(0, 2, 3, 1).cpu()[0, ...], cmap='gray')
    axs[0, 1].set_title('f')
    axs[1, 0].imshow(u.detach().cpu().permute(0, 2, 3, 1)[0, ...], cmap='gray')
    axs[1, 0].set_title('uk')
    axs[1, 1].imshow(torch.abs(f.permute(0, 2, 3, 1) - u.permute(0, 2, 3, 1).detach()).cpu()[0, ...],
                     cmap='gray')
    axs[1, 1].set_title('dif')
    plt.show()
    return outputs

if __name__ == '__main__':
    from PIL import Image
    # import requests

    url = sys.argv[1]
    results_folder = sys.argv[2]
    
    # im = Image.open(requests.get(url, stream=True).raw)
    im = Image.open(url)
    im = crop_image(np.array(im))
    # im = np.array(im)[200:350, 200:350, :]
    org = im / 255.0

    sigma = float(sys.argv[3])

    f = add_rician_noise(org, sigma, kspace=False)
    # f = add_rician_noise(org, sigma)
    # f = add_gaussian_noise(org, sigma)

    lda = sigma / 4
    dip_rician_outs = dip_rician(f=f, sigma=sigma, lr=1e-3,
                                 outter_nits=3400, dev='cuda', org=org,
                                 plot=None,
                                 lda=None, # TV
                                 loss_fn=sys.argv[4]
                                 )
    #
    plt.subplots_adjust(wspace=0, hspace=0)

    fig, axs = plt.subplots(2, 2, figsize=(3*2, 3*2))
    axs[0, 0].imshow(dip_rician_outs['f'], cmap='gray')
    axs[0, 0].set_title('f')
    axs[0, 1].imshow(dip_rician_outs['best_u'])
    axs[0, 1].set_title('best u  - psnr: {:.3f}'.format(max(dip_rician_outs['psnr'])))
    axs[1, 0].plot(dip_rician_outs['loss'])
    axs[1, 1].plot(dip_rician_outs['psnr'])

    # plt.savefig('Gaussian_noise_rician_fidelity.png', bbox_inches='tight')
    plt.show()

    from PIL import Image

    Image.fromarray(np.uint8(dip_rician_outs['best_u'].numpy() * 255)).save(f"{results_folder}/best_u.png")

    mask = f**2 >= 2*sigma**2
    Image.fromarray(np.uint8(mask * 255)).save("mask.png")

    df = pd.DataFrame({ 'loss': dip_rician_outs['loss'], 
                       'psnr_log': dip_rician_outs['psnr'] })
    df.to_csv(f'{results_folder}/execution_log.csv', index=False)

# Execution example:
# python denoise_image_dip_rician.py "natural_images_selection/img_0" "results/natural_images/image_0/rician_norm_fidelity/std_0.25" "0.25" "rician" &
# python denoise_image_dip_rician.py "natural_images_selection/img_0" "results/natural_images/image_0/rician_MSE/std_0.25" "0.25" "mse" &
