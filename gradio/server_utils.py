import sys
sys.path.append("..")
import numpy as np
from _utils import add_gaussian_noise
import matplotlib.pyplot as plt

def get_noisy_image(image, std):
    image = image.astype('float32') / 255
    return np.clip(add_gaussian_noise(image, 0, float(std)), 0., 1.)

def get_figure_plot(loss_log, psnr_log, ssim_log, std):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(loss_log, label="MSE")
    axes[0].scatter(len(loss_log) - 1, loss_log[-1], c="red")
    axes[0].axhline(std**2, linewidth=.8, label="stopping criterion", c="orange")
    axes[0].legend()
    axes[1].plot(psnr_log, label="PSNR")
    axes[1].scatter(len(psnr_log) - 1, psnr_log[-1], c="red")
    axes[1].legend()
    axes[2].plot(ssim_log, label="SSIM")
    axes[2].scatter(len(ssim_log) - 1, ssim_log[-1], c="red")
    axes[2].legend()

    return fig
        

def denoise_image(image, std, max_iterations, use_stopping_criterion, learning_rate, seed_noise_std):

    max_iterations = int(max_iterations)
    use_stopping_criterion = bool(use_stopping_criterion)

    from denoising_net import Net, dev
    from _utils import crop_image

    # quality metrics
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

    # utils for clipping
    import numpy as np

    from torch import Tensor, optim, from_numpy

    img_original = crop_image((image.astype('float32') / 255).transpose(2, 0, 1))
    img = from_numpy(add_gaussian_noise(img_original, 0, std))

    NN = Net().to(dev)
    seed_cpu = from_numpy(
        np.random.uniform(low=0, high=0.1, size=(32, img.shape[1], img.shape[2])).astype('float32'))[None, :]
    
    opt = optim.Adam(NN.parameters(), lr=learning_rate)

    noisy_img_dev = img.to(dev)

    def error(img_approx: Tensor) -> Tensor:
        return (noisy_img_dev - img_approx).square().mean()

    def stop(mse: float, std: float) -> bool:
        return mse < std ** 2

    loss_log, psnr_log, ssim_log = [], [], []

    for _ in range(max_iterations):
        opt.zero_grad()

        noisy_seed_dev = add_gaussian_noise(seed_cpu, 0, seed_noise_std).to(dev)

        # this iteration's image and loss
        img_approx = NN(noisy_seed_dev)
        loss = error(img_approx)

        # update gradients
        loss.backward()
        opt.step()
        
        # yield logs
        img_approx_cpu = np.clip(img_approx.cpu().detach()[0].numpy(), 0., 1.0)
        index, img_ssim = ssim(img_original, img_approx_cpu, data_range=1.0, channel_axis=0, full=True)
        loss_log.append(loss.cpu().detach().numpy())
        psnr_log.append(psnr(img_original, img_approx_cpu, data_range=1.0))
        ssim_log.append(index)
        plt.close() # preemptively free memory
        yield (img_approx.cpu().detach()[0].numpy().transpose(1, 2, 0), img_ssim.transpose(1, 2, 0),
               get_figure_plot(loss_log, psnr_log, ssim_log, std))

        if use_stopping_criterion and stop(loss_log[-1], std):
            break
        
