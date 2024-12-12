import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
# from PIL import Image
from PIL import Image, ImageDraw

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


from scipy.ndimage import binary_erosion


def extract_patches(img: Image, source: torch.Tensor, patch_size: int, category: int, zoom_factor: int = 4):
    """
    Extracts two patches of a specified category and returns their enlarged versions.

    Args:
     - img (Image): Original PIL image.
     - source (torch.Tensor): Source tensor with shape (1, 17, 196).
     - patch_size (int): The size of each patch (assuming square patches).
     - category (int): The target category for which to extract patches.
     - zoom_factor (int): Factor by which to enlarge the extracted patches.

    Returns:
     - patch_images (list): List of enlarged patch images from the specified category.
    """
    img_array = np.array(img.convert("RGB"))  # Convert img to numpy array
    source = source.detach().cpu()

    # Convert source to a (1, 196) label map, each token classified by category
    vis = source.argmax(dim=1)

    # Determine the original image dimensions
    h, w, _ = img_array.shape
    ph = h // patch_size
    pw = w // patch_size

    # Locate tokens for the specified category
    category_positions = (vis == category).nonzero(as_tuple=True)[1].tolist()

    if len(category_positions) < 2:
        raise ValueError(f"Not enough patches found in category {category} to extract two examples.")

    # Take the first two positions found for the target category
    pos1, pos2 = category_positions[:2]

    # Convert token positions back to patch coordinates
    pos1_y, pos1_x = divmod(pos1, pw)
    pos2_y, pos2_x = divmod(pos2, pw)

    # Extract and zoom in on the patches
    patch1 = img_array[pos1_y * patch_size:(pos1_y + 1) * patch_size,
             pos1_x * patch_size:(pos1_x + 1) * patch_size]
    patch2 = img_array[pos2_y * patch_size:(pos2_y + 1) * patch_size,
             pos2_x * patch_size:(pos2_x + 1) * patch_size]

    # Convert to PIL Images for zooming
    patch1_img = Image.fromarray(patch1)
    patch2_img = Image.fromarray(patch2)
    patch1_img = patch1_img.resize((patch1_img.width * zoom_factor, patch1_img.height * zoom_factor), Image.NEAREST)
    patch2_img = patch2_img.resize((patch2_img.width * zoom_factor, patch2_img.height * zoom_factor), Image.NEAREST)

    return patch1_img, patch2_img


def extract_and_highlight_patches(img: Image, source: torch.Tensor, patch_size: int, category: int,
                                  zoom_factor: int = 4):
    """
    Extracts two patches of a specified category, returns their enlarged versions, 
    and the original image with these patches highlighted.

    Args:
     - img (Image): Original PIL image.
     - source (torch.Tensor): Source tensor with shape (1, 17, 196).
     - patch_size (int): The size of each patch (assuming square patches).
     - category (int): The target category for which to extract and highlight patches.
     - zoom_factor (int): Factor by which to enlarge the extracted patches.

    Returns:
     - patch_images (list): List containing two enlarged patch images from the specified category.
     - highlighted_img (Image): Original image with highlighted patches.
    """
    img_array = np.array(img.convert("RGB"))  # Convert img to numpy array
    source = source.detach().cpu()

    # Convert source to a (1, 196) label map, each token classified by category
    vis = source.argmax(dim=1)

    # Determine the original image dimensions
    h, w, _ = img_array.shape
    ph = h // patch_size
    pw = w // patch_size

    # Locate tokens for the specified category
    category_positions = (vis == category).nonzero(as_tuple=True)[1].tolist()

    if len(category_positions) < 2:
        raise ValueError(f"Not enough patches found in category {category} to extract two examples.")

    # Take the first two positions found for the target category
    pos1, pos2 = category_positions[:2]

    # Convert token positions back to patch coordinates
    pos1_y, pos1_x = divmod(pos1, pw)
    pos2_y, pos2_x = divmod(pos2, pw)

    # Extract and zoom in on the patches
    patch1 = img_array[pos1_y * patch_size:(pos1_y + 1) * patch_size,
             pos1_x * patch_size:(pos1_x + 1) * patch_size]
    patch2 = img_array[pos2_y * patch_size:(pos2_y + 1) * patch_size]

    # Convert to PIL Images for zooming
    patch1_img = Image.fromarray(patch1).resize((patch_size * zoom_factor, patch_size * zoom_factor), Image.NEAREST)
    patch2_img = Image.fromarray(patch2).resize((patch_size * zoom_factor, patch_size * zoom_factor), Image.NEAREST)

    # Highlight the patches in the original image
    highlighted_img = img.copy()
    draw = ImageDraw.Draw(highlighted_img)
    rect_color = (255, 0, 0)  # Red color for the rectangle outline
    rect_width = 2  # Width of the rectangle border

    # Draw rectangles around the patches in the original image
    draw.rectangle(
        [(pos1_x * patch_size, pos1_y * patch_size),
         ((pos1_x + 1) * patch_size, (pos1_y + 1) * patch_size)],
        outline=rect_color, width=rect_width
    )
    draw.rectangle(
        [(pos2_x * patch_size, pos2_y * patch_size),
         ((pos2_x + 1) * patch_size, (pos2_y + 1) * patch_size)],
        outline=rect_color, width=rect_width
    )

    return patch1_img, patch2_img, highlighted_img


def make_visualization(
        img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True, alpha: float = 0.65,
        color_intensity: float = 1.5
) -> Image:  # 0.65,1.1
    """
    Create a visualization with deeper mask colors.

    Args:
     - img (Image): Original PIL image
     - source (torch.Tensor): Source tensor for visualization
     - patch_size (int): Size of each patch
     - class_token (bool): Whether to ignore the class token
     - alpha (float): Transparency level of the mask (0-1), where 1 is fully opaque.
     - color_intensity (float): Multiplier for mask color intensity to make it deeper.

    Returns:
     - A PIL image the same size as the input, with mask overlay.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    cmap = [(np.array(color) * color_intensity).clip(0, 1) for color in cmap]  # 深化颜色
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image and apply transparency
    vis_img = np.uint8(vis_img * 255)
    vis_img = Image.fromarray(vis_img)
    vis_img = Image.blend(Image.fromarray((img * 255).astype(np.uint8)), vis_img, alpha=alpha)

    return vis_img
