import torch


def load_normalized_image(path: str):
    import torchvision
    return torchvision.io.read_image(path, mode=torchvision.io.image.ImageReadMode.RGB) / 255


def add_gaussian_noise(img, avg: float, std: float):
    import numpy as np
    return img + np.random.normal(avg, std, img.shape).astype('float32')


def plot_simple_image(img: torch.Tensor, file_name: str = "", show=False):
    import matplotlib.pyplot as plt
    img_transposed = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imshow(img_transposed)
    plt.axis('off')
    plt.axis('scaled')
    plt.tight_layout()
    if show:
        plt.show()
    elif file_name != "":
        plt.savefig(f'{file_name}.png', bbox_inches='tight')
    plt.close(fig)
