"""Utility functions for visualizing images in native space"""
import numpy as np
import matplotlib.pyplot as plt

def clip_LPS(img, xyz):
    x, y, z = xyz
    clip_sag = img[x, :, :]
    clip_sag = np.rot90(clip_sag)
    clip_cor = img[:, y, :]
    clip_cor = np.rot90(clip_cor)
    clip_ax = img[:, :, z]
    clip_ax = np.flip(clip_ax, 0)
    clip_ax = np.rot90(clip_ax, 3)
    return (clip_sag, clip_cor, clip_ax)

def clip_LAS(img, xyz):
    x, y , z = xyz
    clip_sag = img[x, :, :]
    clip_sag = np.flip(clip_sag, 0)
    clip_sag = np.rot90(clip_sag)
    clip_cor = img[:, y, :]
    clip_cor = np.rot90(clip_cor)
    clip_ax = img[:, :, z]
    clip_ax = np.rot90(clip_ax)
    return (clip_sag, clip_cor, clip_ax)

def clip_Identity(img, xyz):
    x, y, z = xyz
    clip_x = img[x, :, :]
    clip_y = img[:, y, :]
    clip_z = img[:, :, z]
    return(clip_x, clip_y, clip_z)

def vis_slices(img, xyz=(100,100,100), orientation="LAS", cmap="gray", clip_range=(-1024, 600)):

    if orientation=="LAS":
        clip_sag, clip_cor, clip_ax = clip_LAS(img, xyz)
    elif orientation=="LPS":
        clip_sag, clip_cor, clip_ax = clip_LPS(img, xyz)
    elif orientation=="Identity":
        clip_sag, clip_cor, clip_ax = clip_Identity(img, xyz)

    if len(clip_range) > 0:
        vmin = clip_range[0]
        vmax = clip_range[1]
    else:
        vmax = np.max(img)
        vmin = np.min(img)
    # fig = plt.figure(1)
    f, ax = plt.subplots(1, 3, figsize=(15, 15))
    sag = ax[0].imshow(clip_sag, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    ax[1].imshow(clip_cor, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    ax[2].imshow(clip_ax, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(sag, ax=ax[2], fraction=0.046, pad=0.04)

def vis_sag(img, x, orientation="LAS", cmap="gray", clip_range=(-1024, 600), colorbar=True):
    xyz = (x, 0, 0)
    clip_sag, _, _ = clip_LAS(img, xyz) if orientation=="LAS" else clip_LPS(img, xyz)

    if len(clip_range) > 0:
        vmin = clip_range[0]
        vmax = clip_range[1]
    else:
        vmax = np.max(img)
        vmin = np.min(img)
    f = plt.figure(figsize=(20, 20))
    sag = plt.imshow(clip_sag, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(sag, fraction=0.046, pad=0.04)


def vis_cor(img, y, orientation="LAS", cmap="gray",clip_range=(-1024, 600), colorbar=True):
    xyz = (0, y, 0)
    _, clip_cor, _ = clip_LAS(img, xyz) if orientation == "LAS" else clip_LPS(img, xyz)
    if len(clip_range) > 0:
        vmin = clip_range[0]
        vmax = clip_range[1]
    else:
        vmax = np.max(img)
        vmin = np.min(img)
    f = plt.figure(figsize=(20, 20))
    cor = plt.imshow(clip_cor, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(cor, fraction=0.046, pad=0.04)

def vis_ax(img, z, orientation="LAS", cmap="gray", clip_range=(-1024, 600), colorbar=True):
    xyz = (0, 0, z)
    _, _, clip_ax = clip_LAS(img, xyz) if orientation == "LAS" else clip_LPS(img, xyz)
    if len(clip_range) > 0:
        vmin = clip_range[0]
        vmax = clip_range[1]
    else:
        vmax = np.max(img)
        vmin = np.min(img)
    f = plt.figure(figsize=(20, 20))
    sag = plt.imshow(clip_ax, interpolation='bilinear', cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(sag, fraction=0.046, pad=0.04)

def vis_overlay_slices(img, label, xyz, orientation="LAS", cmap="turbo", clip_range=(-1024, 600), mask_range=(0,5)):
    label = np.where(label == 0, np.nan, label)
    img_clip_sag, img_clip_cor, img_clip_ax = clip_LAS(img, xyz) if orientation == "LAS" else clip_LPS(img, xyz)
    label_clip_sag, label_clip_cor, label_clip_ax = clip_LAS(label, xyz) if orientation == "LAS" else clip_LPS(label,
                                                                                                               xyz)
    if len(clip_range) > 0:
        vmin = clip_range[0]
        vmax = clip_range[1]
    else:
        vmax = np.max(img)
        vmin = np.min(img)
    f, ax = plt.subplots(1, 3, figsize=(15, 15))

    sag = ax[0].imshow(img_clip_sag, interpolation='bilinear', cmap="gray", alpha=1.0, vmin=clip_range[0],
                       vmax=clip_range[1])
    ax[0].imshow(label_clip_sag, interpolation='none', cmap=cmap, alpha=0.5, vmin=mask_range[0], vmax=mask_range[1])
    ax[1].imshow(img_clip_cor, interpolation='bilinear', cmap="gray", alpha=1.0, vmin=clip_range[0], vmax=clip_range[1])
    ax[1].imshow(label_clip_cor, interpolation='none', cmap=cmap, alpha=0.5, vmin=mask_range[0], vmax=mask_range[1])
    ax[2].imshow(img_clip_ax, interpolation='bilinear', cmap="gray", alpha=1.0, vmin=clip_range[0], vmax=clip_range[1])
    ax[2].imshow(label_clip_ax, interpolation='none', cmap=cmap, alpha=0.5, vmin=mask_range[0], vmax=mask_range[1])

def vis_overlay_single_plane(img, label, xyz, plane="axial", orientation="LAS", cmap="turbo", clip_range=(-1024,600), mask_range=(0,5)):
    # convert 0 to np.nan to make mask transparent
    label = np.where(label==0, np.nan, label)

    img_clips = clip_LAS(img, xyz) if orientation == "LAS" else clip_LPS(img, xyz)
    label_clips = clip_LAS(label, xyz) if orientation == "LAS" else clip_LPS(label, xyz)

    i = 0 if plane=="sagittal" else 1 if plane=="coronal" else 2
    img_clip = img_clips[i]
    label_clip = label_clips[i]

    f = plt.figure(figsize=(20, 20))
    plt.imshow(img_clip, interpolation='bilinear', cmap="gray", alpha=0.8, vmin=clip_range[0], vmax=clip_range[1])
    plt.imshow(label_clip, interpolation='none', cmap=cmap, alpha=0.5, vmin=mask_range[0], vmax=mask_range[1])