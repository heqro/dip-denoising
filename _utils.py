import torch
import torchvision

def __load_image__(path: str, mode: torchvision.io.ImageReadMode):
    import torchvision
    return torchvision.io.read_image(path, mode) / 255

def load_rgb_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.RGB)

def load_gray_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.GRAY)

def add_gaussian_noise(img, avg: float, std: float) -> torch.Tensor:
    import numpy as np
    return img + np.random.normal(avg, std, img.shape).astype('float32')


def save_image(img: torch.Tensor, file_name: str = "", show=False):
    import matplotlib.pyplot as plt
    img_transposed = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imsave(fname=file_name, arr=img_transposed, format="png")
    plt.close(fig)
