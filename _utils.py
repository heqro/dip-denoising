import torch
import torchvision
import numpy as np
from numpy import ndarray

def __load_image__(path: str, mode: torchvision.io.ImageReadMode):
    return torchvision.io.read_image(path, mode) / 255

def load_rgb_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.RGB)

def load_gray_image(path: str):
    return __load_image__(path, torchvision.io.image.ImageReadMode.GRAY)

def crop_image(img, d=32):
    new_height = img.shape[1] - img.shape[1] % d
    new_width = img.shape[2] - img.shape[2] % d
    return img[:, :new_height, :new_width]

def add_gaussian_noise(img, avg: float, std: float) -> torch.Tensor:
    return img + np.random.normal(avg, std, img.shape).astype('float32')

def add_rician_noise(img, std: float) -> torch.Tensor:
    x_center, y_center = 0, 0
    sample_size = img.shape[0] * img.shape[1] * img.shape[2]
    distribution = np.random.normal(loc=[x_center, y_center], scale=std, size=(sample_size, 2))

    noise_vector = np.linalg.norm(distribution, axis=1)

    return img + noise_vector.reshape(img.shape).astype('float32')

def print_image(img: ndarray, file_name: str = ""):
    img_transposed = img.transpose(1, 2, 0)
    if file_name != "":
        import imageio
        imageio.imwrite(uri=file_name, im=img_transposed)
    else: # will only work with RGB images
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(img_transposed)
        plt.axis('off')
        plt.show()
        plt.close(fig)
    
