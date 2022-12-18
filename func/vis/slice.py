from matplotlib import pyplot as plt
import numpy as np

def showslice(img, x, y, z, is_gray=False):
    clip_sag = img[x, :, :]
    clip_sag = np.flip(clip_sag, 0)
    clip_sag = np.rot90(clip_sag)
    clip_cor = img[:, y, :]
    clip_cor = np.rot90(clip_cor)
    clip_ax = img[:,:,z]
    clip_ax = np.rot90(clip_ax)
    f, ax = plt.subplots(1,3, figsize=(15,15))
    if is_gray:
        sag = ax[0].imshow(clip_sag, interpolation='nearest', cmap='gray')
        ax[1].imshow(clip_cor, interpolation='nearest', cmap='gray')
        ax[2].imshow(clip_ax, interpolation='nearest', cmap='gray')
    else:
        sag = ax[0].imshow(clip_sag, interpolation='nearest')
        ax[1].imshow(clip_cor, interpolation='nearest')
        ax[2].imshow(clip_ax, interpolation='nearest')
    plt.colorbar(sag, ax=ax[2], fraction=0.046, pad=0.04)