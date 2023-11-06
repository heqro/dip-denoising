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
    new_height = img.shape[0] - img.shape[0] % d
    new_width = img.shape[1] - img.shape[1] % d
    return img[:, :new_height, :new_width]

def add_gaussian_noise(img, avg: float, std: float) -> torch.Tensor:
    return img + np.random.normal(avg, std, img.shape).astype('float32')

# def add_rician_noise(img, std: float) -> torch.Tensor:
#     x_center, y_center = 0, 0
#     sample_size = img.shape[0] * img.shape[1] * img.shape[2]
#     distribution = np.random.normal(loc=[x_center, y_center], scale=std, size=(sample_size, 2))

#     noise_vector = np.linalg.norm(distribution, axis=1)

#     return img + noise_vector.reshape(img.shape).astype('float32')

def add_rician_noise(u, sigma, kspace=False) -> torch.Tensor:
    if kspace:
        j = complex(0, 1)
        j = np.asarray(j)[None, None, None, ...]
        w, h, c = np.shape(u)
        omega = w*h*c
        scale = 1/(np.pi) #TODO justificar
        n = 1
        u_F = np.fft.fftn(u, s=[n*w, n*h], axes=(0, 1))
        f_F = u_F + np.sqrt(omega*scale)*(sigma * np.random.randn(n*w, n*h, c) + j * sigma * np.random.randn(n*w, n*h, c))
        i_f_F = np.fft.ifftn(f_F, s=[n*w, n*h], axes=(0, 1))
        f = np.sqrt(np.real(i_f_F)**2 + np.imag(i_f_F)**2)
        # f = np.real(i_f_F)
        f = f[0:w, 0:h, :]
    else:
        f_real = u + sigma * np.random.randn(u.shape[0], u.shape[1], u.shape[2])
        f_imag = sigma * np.random.randn(u.shape[0], u.shape[1], u.shape[2])
        f = np.sqrt(f_real**2 + f_imag**2)
    return f

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
    
