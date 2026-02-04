import math
import platform
from typing import Optional

import fastai
import matplotlib.colors as mcolors
import numpy as np
import torch
from fastai.torch_core import default_device
from matplotlib import pyplot as plt


def plot_batch(batch, image_num=0, labels: Optional[list[str]] = None, n: int = 1):
    # Load one batch of data
    x, y = batch  # x: images, y: masks

    # Print the shape of the image tensor and the mask
    print(f"Image tensor shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    # force x to float
    x = x.float()

    _, channels, _, _ = x.shape

    if image_num >= x.shape[0]:
        print(f"Image number {image_num} is out of range for this batch.")
        return

    # Clamp the number of images to what exists in the batch
    n = max(1, n)
    n = min(n, x.shape[0] - image_num)

    # Ensure there are at least 3 channels to form an RGB image
    if channels < 3:
        print("There are less than 3 channels available. Cannot form an RGB image.")
        return

    num_cols = channels + 3  # RGB + per-channel + label + overlay
    fig, axs = plt.subplots(
        n, num_cols, figsize=(30, 4.5 * n), gridspec_kw={"hspace": 0.15}
    )
    # Ensure axs is always 2D for consistent indexing
    if n == 1:
        axs = np.expand_dims(axs, axis=0)

    for row in range(n):
        img_idx = image_num + row

        # Extract the first three channels to form an RGB image
        rgb_img = x[img_idx, :3].cpu().numpy()
        rgb_img = np.transpose(
            rgb_img, (1, 2, 0)
        )  # Rearrange dimensions to height x width x channels
        rgb_img = (rgb_img - rgb_img.min()) / (
            rgb_img.max() - rgb_img.min()
        )  # Normalize to [0, 1] for displaying

        # Display the RGB image
        axs[row, 0].imshow(rgb_img)
        if row == 0:
            if labels:
                axs[row, 0].set_title(labels[0])
            else:
                axs[row, 0].set_title(f"RGB Image #{img_idx}")
        axs[row, 0].axis("off")

        # Plot each channel for the specified image number
        for ch in range(channels):
            img_channel = x[img_idx, ch].cpu().numpy()

            axs[row, ch + 1].imshow(img_channel, cmap="gray")
            if row == 0:
                if labels:
                    axs[row, ch + 1].set_title(labels[ch + 1])
                else:
                    axs[row, ch + 1].set_title(f"Channel {ch + 1}")
            axs[row, ch + 1].axis("off")
    colors = [
        "#8B4513",
        "white",
        "#808080",
        "black",
        "pink",
    ]  # brown, white, grey, black
    n_bins = 5
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "brown_white_grey_black", colors, N=n_bins
    )
    # remap 99 values to 5 for visualization
    # Plot the label mask for each requested image number
    for row in range(n):
        img_idx = image_num + row
        mask = y[img_idx].clone()  # avoid mutating the batch tensor
        mask[mask == 99] = 5
        mask_np = mask.cpu().numpy()

        # Label-only view
        axs[row, -2].imshow(
            mask_np,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=4,
        )
        if row == 0:
            axs[row, -2].set_title("Label")
        axs[row, -2].axis("off")

        # Overlay mask on RGB
        rgb_overlay = x[img_idx, :3].cpu().numpy()
        rgb_overlay = np.transpose(rgb_overlay, (1, 2, 0))
        rgb_overlay = (rgb_overlay - rgb_overlay.min()) / (
            rgb_overlay.max() - rgb_overlay.min()
        )
        axs[row, -1].imshow(rgb_overlay, cmap=None)
        axs[row, -1].imshow(
            mask_np,
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=4,
            alpha=0.4,
        )
        if row == 0:
            axs[row, -1].set_title("Overlay")
        axs[row, -1].axis("off")

    plt.show()


def show_histo(batch, image_num=0, labels: Optional[list[str]] = None):
    tensor = batch[0][image_num].cpu().float()
    num_channels = tensor.shape[0]

    # Calculate the number of rows and columns for subplots
    # aiming for a square-ish layout
    num_rows = math.ceil(math.sqrt(num_channels))
    num_cols = math.ceil(num_channels / num_rows)

    # Create a figure with dynamic subplots
    _, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    # Ensure axs is always a 2D array for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1 or num_cols == 1:
        axs = axs.reshape(num_rows, num_cols)

    # Flatten each band and plot its histogram
    for i in range(num_channels):
        values = tensor[i].flatten().numpy()  # Convert to NumPy array for plotting
        row, col = divmod(i, num_cols)

        ax = axs[row, col]
        ax.hist(values, bins=100, alpha=0.75)
        if labels:
            ax.set_title(labels[i])
        else:
            ax.set_title(f"Band {i + 1} Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.set_xlim(-2, 2)

    # Adjust layout to prevent overlap and hide empty subplots
    for ax in axs.flatten()[num_channels:]:
        ax.set_visible(False)  # Hide unused subplots

    plt.tight_layout()
    plt.show()


def print_system_info():
    # Gather information
    info = {
        "PyTorch Version": torch.__version__,
        "CUDA Available": "Yes" if torch.cuda.is_available() else "No",
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",  # type: ignore
        "Python Version": platform.python_version(),
        "Fastai Version": fastai.__version__,
        "Default Device": default_device(),
        "Device Name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "N/A",
    }

    # Find the maximum key length for alignment
    max_key_length = max(len(key) for key in info.keys())

    # Print the table
    print("System Information")
    print("-" * (max_key_length + 20))  # Adjusting based on expected value lengths
    for key, value in info.items():
        print(f"{key.ljust(max_key_length)} : {value}")
    print("-" * (max_key_length + 20))
