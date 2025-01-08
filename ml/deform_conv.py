"""
Implements deformable conv operation in "https://arxiv.org/abs/1703.06211".
"""

import numpy as np


def bi_interp(M, x, y):
    """Perform bilinear interpolation for the given x and y coordinates
    on the M.

    Args:
        M (numpy.ndarray): Input feature map of shape (C, H, W).
        x (float): X-coordinate(s) for interpolation.
        y (float): Y-coordinate(s) for interpolation.

    Returns:
        numpy.ndarray: Interpolated values of shape (C, ).
    """
    C, H, W = M.shape

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    ia = M[:, y0, x0]
    ib = M[:, y1, x0]
    ic = M[:, y0, x1]
    id = M[:, y1, x1]

    return wa * ia + wb * ib + wc * ic + wd * id


def deform_conv2d(M, weights, offsets, stride=1, padding=1):
    """Perform deformable convolution on the input feature map.

    Args:
        M (numpy.ndarray): Input feature map of shape (C_in, H_in, W_in).
        weights (numpy.ndarray): Convolution weights of shape (C_out, C_in, kH, kW).
        offsets (numpy.ndarray): Offset map of shape (2 * kH * kW, H_out, W_out).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input feature map.

    Returns:
        numpy.ndarray: Output feature map of shape (C_out, H_out, W_out).
    """
    C_in, H_in, W_in = M.shape
    C_out, _, kH, kW = weights.shape

    # Calculate output dimensions
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    # Pad the input feature map
    padded_input = np.pad(
        M, ((0, 0), (padding, padding), (padding, padding)), mode="constant"
    )

    # Initialize output feature map
    output = np.zeros((C_out, H_out, W_out))

    # Iterate over each output spatial location
    for out_y in range(H_out):
        for out_x in range(W_out):
            # Calculate the starting point on the input feature map
            in_y = out_y * stride
            in_x = out_x * stride

            # Iterate over each convolution kernel position
            for k_y in range(kH):
                for k_x in range(kW):
                    # Compute the index for the offset map
                    offset_idx = 2 * (k_y * kW + k_x)

                    # Get the offset values
                    delta_y = offsets[offset_idx, out_y, out_x]
                    delta_x = offsets[offset_idx + 1, out_y, out_x]

                    # Calculate the sampling position with offset
                    sample_y = in_y + k_y + delta_y
                    sample_x = in_x + k_x + delta_x

                    # Perform bilinear interpolation
                    sampled_feat = bi_interp(padded_input, sample_x, sample_y)

                    # Accumulate the weighted sum for each output channel
                    for c_out in range(C_out):
                        output[c_out, out_y, out_x] += weights[c_out, :, k_y, k_x].dot(
                            sampled_feat
                        )

    return output


def deformable_roi_pooling(M, rois, offsets, output_size=(7, 7), spatial_scale=1.0):
    """Perform deformable ROI pooling on the input feature map.

    Args:
        M (numpy.ndarray): Input feature map of shape (C, H, W).
        rois (numpy.ndarray): ROIs of shape (N, 5), where each ROI is (batch_idx, x1, y1, x2, y2).
        offsets (numpy.ndarray): Offset map of shape (2 * output_size[0] * output_size[1], N, output_size[0], output_size[1]).
        output_size (tuple): The height and width of the output feature map (e.g., (7, 7)).
        spatial_scale (float): Scale factor to map ROI coordinates to feature map scale.

    Returns:
        numpy.ndarray: Output feature maps of shape (C, N, output_size[0], output_size[1]).
    """
    C, H, W = M.shape
    N = rois.shape[0]
    out_h, out_w = output_size

    # Initialize output feature map
    output = np.zeros((C, N, out_h, out_w), dtype=M.dtype)

    for n in range(N):
        roi = rois[n]
        batch_idx, x1, y1, x2, y2 = roi

        # Scale ROI coordinates
        x1 *= spatial_scale
        y1 *= spatial_scale
        x2 *= spatial_scale
        y2 *= spatial_scale

        # Compute the width and height of the ROI
        roi_width = max(x2 - x1, 1.0)
        roi_height = max(y2 - y1, 1.0)

        # Compute the size of each bin
        bin_size_w = roi_width / out_w
        bin_size_h = roi_height / out_h

        for ph in range(out_h):
            for pw in range(out_w):
                # Compute the start of the bin
                bin_start_x = x1 + pw * bin_size_w
                bin_start_y = y1 + ph * bin_size_h

                # Compute the center of the bin
                bin_center_x = bin_start_x + bin_size_w / 2
                bin_center_y = bin_start_y + bin_size_h / 2

                # Compute the offsets for this bin
                offset_idx = 2 * (ph * out_w + pw)
                delta_y = offsets[offset_idx, n, ph, pw]
                delta_x = offsets[offset_idx + 1, n, ph, pw]

                # Apply offsets
                sample_x = bin_center_x + delta_x * bin_size_w
                sample_y = bin_center_y + delta_y * bin_size_h

                # Perform bilinear interpolation
                sampled_feat = bi_interp(M, sample_x, sample_y)  # Shape: (C,)

                # Assign to output
                output[:, n, ph, pw] = sampled_feat

    return output


def run_conv2d():
    # Example input feature map: 1 channel, 5x5 spatial dimensions
    C_in, H_in, W_in = 1, 5, 5
    input_feature = (
        np.arange(C_in * H_in * W_in).reshape(C_in, H_in, W_in).astype(np.float32)
    )

    # Example convolution weights: 1 output channel, 1 input channel, 3x3 kernel
    C_out, _, kH, kW = 1, C_in, 3, 3
    weights = np.ones((C_out, C_in, kH, kW), dtype=np.float32)

    # Example offsets: 2 * 3 * 3 = 18 offsets, output size 3x3 (assuming stride=1, padding=1)
    H_out, W_out = 5, 5
    offsets = np.zeros((2 * kH * kW, H_out, W_out), dtype=np.float32)

    # For demonstration, let's add a simple offset that shifts sampling by (0.5, 0.5) for all kernel positions
    for i in range(2 * kH * kW):
        offsets[i, :, :] = 0.5 if i % 2 else 0.5  # delta_y and delta_x

    # Perform deformable convolution
    output = deform_conv2d(input_feature, weights, offsets, stride=1, padding=1)

    print("Input Feature Map:", input_feature[0])
    print("Offsets:", offsets)
    print("Output Feature Map:", output[0])


def run_roi_pool():
    # Example input feature map: 3 channels, 10x10 spatial dimensions
    C, H, W = 3, 10, 10
    np.random.seed(0)  # For reproducibility
    feature_map = np.random.rand(C, H, W).astype(np.float32)

    # Example ROIs: 2 ROIs
    # Format: (batch_idx, x1, y1, x2, y2)
    rois = np.array([[0, 1, 1, 8, 8], [0, 2, 2, 6, 6]], dtype=np.float32)

    N = rois.shape[0]
    out_h, out_w = 3, 3  # Output size for each ROI

    # Example offsets: 2 * 3 * 3 = 18 offsets per ROI
    offsets = np.zeros((2 * out_h * out_w, N, out_h, out_w), dtype=np.float32)

    # For demonstration, let's add random small offsets
    # Normally, these offsets would be learned during training
    offsets = np.random.uniform(-0.5, 0.5, size=offsets.shape).astype(np.float32)

    # Perform deformable ROI pooling
    output = deformable_roi_pooling(
        feature_map, rois, offsets, output_size=(out_h, out_w), spatial_scale=1.0
    )

    print("Input Feature Map Shape:", feature_map.shape)
    print("ROIs:", rois)
    print("Offsets Shape:", offsets.shape)
    print("Output Feature Maps Shape:", output.shape)
    print("Output Feature Maps:", output)


if __name__ == "__main__":
    run_conv2d()
    run_roi_pool()
