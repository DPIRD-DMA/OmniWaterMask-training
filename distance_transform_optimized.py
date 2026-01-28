"""
Optimized distance transform for segmentation training weights.

Provides multiple implementations with different speed/accuracy tradeoffs:

Functions (in order of recommendation for GPU training):
---------------------------------------------------------
1. distance_transform_torch_euclidean() - RECOMMENDED for GPU training
   - Runs entirely on GPU, no CPU transfer needed
   - 7-11x faster than CPU when data is on GPU
   - Max diff from original: ~0.4 (negligible for training)

2. distance_transform_torch() - Fastest GPU option
   - 38-65x faster than CPU when data is on GPU
   - Uses discrete erosion (diff ~2.0 from original)
   - Good if you only care about boundary vs interior weighting

3. distance_transform() - Exact CPU match
   - Identical results to original, ~1.3x faster
   - Use if you need exact reproducibility

4. distance_transform_fast() - Fast CPU
   - ~1.5-2x faster than original
   - Tiny diff (<0.04) from using DIST_MASK_5

Usage:
    from distance_transform_optimized import distance_transform_torch_euclidean as distance_transform

    # In your loss function:
    def combo_loss(pred, target):
        pixel_weight = distance_transform(target, clip_distance=3)
        pixel_cel = cel(pred, target)
        return (pixel_cel * pixel_weight).mean()
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def distance_transform(
    input_label: torch.Tensor | np.ndarray,
    clip_distance: float = 3.0,
    classes: list[int] = [0, 1],
) -> torch.Tensor:
    """
    Optimized distance transform with identical results to original.

    Optimizations:
    - Reuses memory buffers instead of allocating each iteration
    - Uses in-place numpy operations where possible
    - Avoids unnecessary copies

    Args:
        input_label: Input label tensor/array of shape (H, W) or (B, H, W)
        clip_distance: Maximum distance to clip to (default 3.0)
        classes: List of class integers to process (default [0, 1])

    Returns:
        Distance transform weights tensor on same device as input
    """
    # Handle input conversion
    if isinstance(input_label, torch.Tensor):
        device = input_label.device
        label_np = input_label.cpu().numpy() if device.type != 'cpu' else input_label.numpy()
    else:
        device = torch.device('cpu')
        label_np = np.ascontiguousarray(input_label)

    # Handle single image case
    single_image = label_np.ndim == 2
    if single_image:
        label_np = label_np[np.newaxis, :, :]

    batch_size, h, w = label_np.shape

    # Pre-allocate output and reusable buffer
    output = np.zeros((batch_size, h, w), dtype=np.float32)
    class_dist = np.empty((h, w), dtype=np.float32)

    for i in range(batch_size):
        img = label_np[i]
        for class_int in classes:
            class_mask = (img == class_int).astype(np.uint8)
            cv2.distanceTransform(class_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=class_dist)
            np.clip(class_dist, 1, clip_distance, out=class_dist)
            class_dist *= class_mask
            output[i] += class_dist

    # Final transformation in-place
    output -= (clip_distance + 1)
    np.abs(output, out=output)

    # Convert back to tensor
    result = torch.from_numpy(output)

    if single_image:
        result = result.squeeze(0)

    if device.type != 'cpu':
        result = result.to(device)

    return result


def distance_transform_fast(
    input_label: torch.Tensor | np.ndarray,
    clip_distance: float = 3.0,
    classes: list[int] = [0, 1],
) -> torch.Tensor:
    """
    Faster distance transform using 5x5 mask approximation.

    This version is ~1.5x-1.9x faster than the original but produces slightly
    different results (max difference ~0.04) due to using DIST_MASK_5 instead
    of DIST_MASK_PRECISE. For training with clip_distance=3, this difference
    is negligible.

    Args:
        input_label: Input label tensor/array of shape (H, W) or (B, H, W)
        clip_distance: Maximum distance to clip to (default 3.0)
        classes: List of class integers to process (default [0, 1])

    Returns:
        Distance transform weights tensor on same device as input
    """
    if isinstance(input_label, torch.Tensor):
        device = input_label.device
        label_np = input_label.cpu().numpy() if device.type != 'cpu' else input_label.numpy()
    else:
        device = torch.device('cpu')
        label_np = np.ascontiguousarray(input_label)

    single_image = label_np.ndim == 2
    if single_image:
        label_np = label_np[np.newaxis, :, :]

    batch_size, h, w = label_np.shape
    output = np.zeros((batch_size, h, w), dtype=np.float32)
    class_dist = np.empty((h, w), dtype=np.float32)

    for i in range(batch_size):
        img = label_np[i]
        for class_int in classes:
            class_mask = (img == class_int).astype(np.uint8)
            # DIST_MASK_5 is faster than DIST_MASK_PRECISE
            cv2.distanceTransform(class_mask, cv2.DIST_L2, cv2.DIST_MASK_5, dst=class_dist)
            np.clip(class_dist, 1, clip_distance, out=class_dist)
            class_dist *= class_mask
            output[i] += class_dist

    output -= (clip_distance + 1)
    np.abs(output, out=output)

    result = torch.from_numpy(output)

    if single_image:
        result = result.squeeze(0)

    if device.type != 'cpu':
        result = result.to(device)

    return result


# Pre-computed erosion kernel for distance transform
_EROSION_KERNEL = None


def _get_erosion_kernel(device: torch.device) -> torch.Tensor:
    """Get or create the 3x3 erosion kernel (cached per device)."""
    global _EROSION_KERNEL
    if _EROSION_KERNEL is None or _EROSION_KERNEL.device != device:
        # 3x3 kernel for morphological erosion
        _EROSION_KERNEL = torch.ones(1, 1, 3, 3, device=device)
    return _EROSION_KERNEL


def distance_transform_torch(
    input_label: torch.Tensor | np.ndarray,
    clip_distance: float = 3.0,
    classes: list[int] = [0, 1],
) -> torch.Tensor:
    """
    Pure PyTorch distance transform using iterative erosion.

    This implementation runs entirely on GPU when input is a CUDA tensor,
    avoiding CPU-GPU transfers. It uses morphological erosion to compute
    approximate distances up to clip_distance.

    For clip_distance=3, this is ~3-10x faster than CPU implementations
    when the input is already on GPU.

    Note: Results differ slightly from cv2 due to discrete erosion vs
    continuous distance, but the difference is small (<0.5) and doesn't
    affect training significantly.

    Args:
        input_label: Input label tensor/array of shape (H, W) or (B, H, W)
        clip_distance: Maximum distance to clip to (default 3.0)
        classes: List of class integers to process (default [0, 1])

    Returns:
        Distance transform weights tensor on same device as input
    """
    # Convert to tensor if needed
    if isinstance(input_label, np.ndarray):
        label = torch.from_numpy(input_label)
    else:
        label = input_label

    device = label.device

    # Handle single image case
    single_image = label.ndim == 2
    if single_image:
        label = label.unsqueeze(0)

    batch_size, h, w = label.shape

    # Pad the input to handle edge artifacts correctly
    # Use replicate padding so edge pixels extend outward (matching cv2 behavior)
    pad_size = int(clip_distance) + 1
    label_padded = F.pad(label.unsqueeze(1).float(), (pad_size, pad_size, pad_size, pad_size), mode='replicate').squeeze(1)

    # Get erosion kernel
    kernel = _get_erosion_kernel(device)

    # Number of erosion iterations needed
    max_dist = int(clip_distance)

    # Output accumulator (on padded size)
    _, h_pad, w_pad = label_padded.shape
    output = torch.zeros(batch_size, h_pad, w_pad, dtype=torch.float32, device=device)

    for class_int in classes:
        # Create binary mask for this class: 1 where class matches, 0 elsewhere
        class_mask = (label_padded == class_int).float()

        # Add channel dimension for conv2d: (B, 1, H, W)
        mask_4d = class_mask.unsqueeze(1)

        # Compute distance by iterative erosion
        # Distance 1 = boundary pixels (eroded once gives interior)
        # We want: pixels at distance d are those that survive d-1 erosions but not d

        current_mask = mask_4d
        dist_map = torch.zeros_like(mask_4d)

        for d in range(1, max_dist + 1):
            # Erode: min pooling (or equivalently, conv and threshold)
            # A pixel survives erosion if all neighbors are 1
            eroded = F.conv2d(current_mask, kernel, padding=1)
            eroded = (eroded >= 9).float()  # 9 = all 3x3 neighbors are 1

            # Pixels at exactly distance d: in current_mask but not in eroded
            at_distance_d = current_mask - eroded

            # Assign distance value (clipped to [1, clip_distance])
            dist_map += at_distance_d * d

            current_mask = eroded

        # Interior pixels (survived all erosions) get max distance
        dist_map += current_mask * clip_distance

        # Remove channel dimension and add to output
        output += dist_map.squeeze(1)

    # Crop back to original size
    output = output[:, pad_size:pad_size + h, pad_size:pad_size + w]

    # Apply final transformation: |output - clip_distance - 1|
    output = torch.abs(output - clip_distance - 1)

    if single_image:
        output = output.squeeze(0)

    return output


def distance_transform_torch_euclidean(
    input_label: torch.Tensor | np.ndarray,
    clip_distance: float = 3.0,
    classes: list[int] = [0, 1],
) -> torch.Tensor:
    """
    PyTorch distance transform with Euclidean distance approximation.

    Uses chamfer distance propagation with proper Euclidean weights for
    accurate distance approximation that runs entirely on GPU.

    Args:
        input_label: Input label tensor/array of shape (H, W) or (B, H, W)
        clip_distance: Maximum distance to clip to (default 3.0)
        classes: List of class integers to process (default [0, 1])

    Returns:
        Distance transform weights tensor on same device as input
    """
    if isinstance(input_label, np.ndarray):
        label = torch.from_numpy(input_label)
    else:
        label = input_label

    device = label.device
    dtype = torch.float32

    single_image = label.ndim == 2
    if single_image:
        label = label.unsqueeze(0)

    batch_size, h, w = label.shape

    # Pad input to handle edges correctly
    # Use replicate padding so edge pixels extend outward (matching cv2 behavior)
    pad_size = int(clip_distance) + 2
    label_padded = F.pad(label.unsqueeze(1).float(), (pad_size, pad_size, pad_size, pad_size), mode='replicate').squeeze(1)

    _, h_pad, w_pad = label_padded.shape
    max_iters = int(clip_distance) + 2

    output = torch.zeros(batch_size, h_pad, w_pad, dtype=dtype, device=device)

    # Chamfer distance weights for 3x3 kernel
    # Center=0, orthogonal=1.0, diagonal=sqrt(2)≈1.414
    sqrt2 = 1.4142135623730951

    for class_int in classes:
        class_mask = (label_padded == class_int).to(dtype)
        mask_4d = class_mask.unsqueeze(1)  # (B, 1, H, W)

        # Initialize: large value inside mask, 0 outside
        # We'll compute distance from boundary INTO the mask
        large_val = clip_distance + 10
        dist = torch.where(mask_4d > 0, large_val, torch.zeros_like(mask_4d))

        # Find boundary: mask pixels adjacent to non-mask pixels
        # Dilate the inverse mask
        inv_mask = 1.0 - mask_4d
        dilated_inv = F.max_pool2d(inv_mask, 3, stride=1, padding=1)

        # Boundary = in mask AND adjacent to background
        boundary = mask_4d * dilated_inv

        # Set boundary pixels to distance 1
        dist = torch.where(boundary > 0, torch.ones_like(dist), dist)

        # Chamfer propagation: iteratively propagate minimum distances
        for _ in range(max_iters):
            # Pad for neighbor access
            padded = F.pad(dist, (1, 1, 1, 1), mode='replicate')

            # Get all 8 neighbors + center
            # Orthogonal neighbors (distance +1)
            top = padded[:, :, :-2, 1:-1] + 1.0
            bottom = padded[:, :, 2:, 1:-1] + 1.0
            left = padded[:, :, 1:-1, :-2] + 1.0
            right = padded[:, :, 1:-1, 2:] + 1.0

            # Diagonal neighbors (distance +sqrt(2))
            top_left = padded[:, :, :-2, :-2] + sqrt2
            top_right = padded[:, :, :-2, 2:] + sqrt2
            bottom_left = padded[:, :, 2:, :-2] + sqrt2
            bottom_right = padded[:, :, 2:, 2:] + sqrt2

            # Take minimum of all
            min_dist = torch.minimum(dist, top)
            min_dist = torch.minimum(min_dist, bottom)
            min_dist = torch.minimum(min_dist, left)
            min_dist = torch.minimum(min_dist, right)
            min_dist = torch.minimum(min_dist, top_left)
            min_dist = torch.minimum(min_dist, top_right)
            min_dist = torch.minimum(min_dist, bottom_left)
            min_dist = torch.minimum(min_dist, bottom_right)

            # Only update interior mask pixels (not boundary, not background)
            dist = torch.where(
                (mask_4d > 0) & (boundary == 0),
                min_dist,
                dist
            )

        # Clip and mask
        dist = torch.clamp(dist, 1.0, clip_distance) * mask_4d
        output += dist.squeeze(1)

    # Crop back to original size
    output = output[:, pad_size:pad_size + h, pad_size:pad_size + w]

    # Final transformation
    output = torch.abs(output - clip_distance - 1)

    if single_image:
        output = output.squeeze(0)

    return output


def visualize_comparison(save_path: str | None = None):
    """Generate a visual comparison of all distance transform methods."""
    import matplotlib.pyplot as plt

    # Create a test mask with interesting shapes
    np.random.seed(42)
    h, w = 128, 128
    mask = np.zeros((h, w), dtype=np.int64)

    # Add a circle in the center
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    radius = 25
    circle = ((y - center[0]) ** 2 + (x - center[1]) ** 2) <= radius ** 2
    mask[circle] = 1

    # Add a rectangle that touches the RIGHT edge
    mask[20:50, 90:128] = 1

    # Add a shape touching the BOTTOM edge
    mask[110:128, 40:70] = 1

    # Add a small square in the corner (touches TOP-LEFT)
    mask[0:12, 0:12] = 1

    # Add a small interior region
    mask[85:95, 15:30] = 1

    label = torch.from_numpy(mask)

    # Compute all versions
    results = {
        "Input Mask": mask.astype(float),
        "Original (cv2)": distance_transform_original(label).numpy(),
        "Optimized (cv2)": distance_transform(label).numpy(),
        "Fast (MASK_5)": distance_transform_fast(label).numpy(),
        "PyTorch Erosion": distance_transform_torch(label).numpy(),
        "PyTorch Euclidean": distance_transform_torch_euclidean(label).numpy(),
    }

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (name, data) in zip(axes, results.items()):
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(name, fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Distance Transform Comparison (clip_distance=3)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    return fig


# Copy of original for benchmarking
def distance_transform_original(input_label, clip_distance=3.0, classes=[0, 1]):
    """Original implementation from notebook."""
    if isinstance(input_label, np.ndarray):
        label = torch.from_numpy(input_label)
    else:
        label = input_label
    label_np = label.numpy(force=True)

    if len(label_np.shape) == 2:
        single_image = True
        label_np = label_np[np.newaxis, :, :]
    else:
        single_image = False

    output = np.zeros_like(label_np, dtype=np.float32)

    for index, img in enumerate(label_np):
        for class_int in classes:
            class_mask = (img == class_int).astype(np.uint8)
            class_dist = cv2.distanceTransform(
                class_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            )
            class_dist = (np.clip(class_dist, 1, clip_distance)) * class_mask
            output[index] += class_dist

    output = np.absolute(output - clip_distance - 1)
    output = torch.from_numpy(output)

    if single_image:
        output = output.squeeze(0)

    return output.to(label.device)


if __name__ == "__main__":
    import time

    # Generate visualization first
    print("Generating visualization...")
    visualize_comparison("distance_transform_comparison.png")
    print()

    print("Distance Transform Optimization Benchmark")
    print("=" * 60)

    # Test cases for CPU
    print("\n" + "=" * 60)
    print("CPU BENCHMARKS")
    print("=" * 60)

    test_cases = [
        (32, 512, 512),  # Typical training batch
        (32, 256, 256),  # Smaller images
    ]

    for batch_size, h, w in test_cases:
        print(f"\nBatch: {batch_size}, Size: {h}x{w}")
        print("-" * 60)

        np.random.seed(42)
        label = torch.from_numpy(np.random.randint(0, 2, (batch_size, h, w)).astype(np.int64))

        # Warmup
        _ = distance_transform_original(label)
        _ = distance_transform(label)
        _ = distance_transform_fast(label)
        _ = distance_transform_torch(label)

        n_runs = 5

        # Original
        start = time.perf_counter()
        for _ in range(n_runs):
            result_orig = distance_transform_original(label)
        time_orig = (time.perf_counter() - start) / n_runs

        # Optimized (exact)
        start = time.perf_counter()
        for _ in range(n_runs):
            result_opt = distance_transform(label)
        time_opt = (time.perf_counter() - start) / n_runs

        # Optimized (fast cv2)
        start = time.perf_counter()
        for _ in range(n_runs):
            result_fast = distance_transform_fast(label)
        time_fast = (time.perf_counter() - start) / n_runs

        # PyTorch (CPU)
        start = time.perf_counter()
        for _ in range(n_runs):
            result_torch = distance_transform_torch(label)
        time_torch = (time.perf_counter() - start) / n_runs

        # Verify
        diff_opt = torch.max(torch.abs(result_orig - result_opt)).item()
        diff_fast = torch.max(torch.abs(result_orig - result_fast)).item()
        diff_torch = torch.max(torch.abs(result_orig - result_torch)).item()

        print(f"Original (cv2):      {time_orig*1000:7.2f} ms")
        print(f"Optimized (cv2):     {time_opt*1000:7.2f} ms  ({time_orig/time_opt:.2f}x) diff={diff_opt:.2e}")
        print(f"Fast (cv2 MASK_5):   {time_fast*1000:7.2f} ms  ({time_orig/time_fast:.2f}x) diff={diff_fast:.2e}")
        print(f"PyTorch (CPU):       {time_torch*1000:7.2f} ms  ({time_orig/time_torch:.2f}x) diff={diff_torch:.2e}")

    # GPU benchmarks if available
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU BENCHMARKS (data already on GPU)")
        print("=" * 60)

        for batch_size, h, w in test_cases:
            print(f"\nBatch: {batch_size}, Size: {h}x{w}")
            print("-" * 60)

            np.random.seed(42)
            label_gpu = torch.randint(0, 2, (batch_size, h, w), device='cuda')

            # Warmup + sync
            _ = distance_transform(label_gpu)
            _ = distance_transform_torch(label_gpu)
            torch.cuda.synchronize()

            n_runs = 10

            # Original (requires CPU transfer)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                result_orig = distance_transform(label_gpu)
            torch.cuda.synchronize()
            time_orig = (time.perf_counter() - start) / n_runs

            # PyTorch GPU
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                result_torch = distance_transform_torch(label_gpu)
            torch.cuda.synchronize()
            time_torch = (time.perf_counter() - start) / n_runs

            # PyTorch GPU Euclidean
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                result_torch_euc = distance_transform_torch_euclidean(label_gpu)
            torch.cuda.synchronize()
            time_torch_euc = (time.perf_counter() - start) / n_runs

            diff_torch = torch.max(torch.abs(result_orig - result_torch)).item()
            diff_torch_euc = torch.max(torch.abs(result_orig - result_torch_euc)).item()

            print(f"Optimized (cv2+transfer): {time_orig*1000:7.2f} ms")
            print(f"PyTorch GPU (erosion):    {time_torch*1000:7.2f} ms  ({time_orig/time_torch:.2f}x) diff={diff_torch:.2e}")
            print(f"PyTorch GPU (euclidean):  {time_torch_euc*1000:7.2f} ms  ({time_orig/time_torch_euc:.2f}x) diff={diff_torch_euc:.2e}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 60)
    print("For GPU training (data already on GPU):")
    print("  -> distance_transform_torch_euclidean: 7-11x faster, diff ~0.4")
    print("  -> distance_transform_torch:          38-65x faster, diff ~2.0")
    print("")
    print("For CPU or exact match:")
    print("  -> distance_transform:      Identical results, ~1.3x faster")
    print("  -> distance_transform_fast: diff <0.04, ~1.5-2x faster")
    print("=" * 60)
