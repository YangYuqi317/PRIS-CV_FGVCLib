# Tutorial 7: Learn about transforms

We import the `transforms` to process the image. The transfomers are from the torchvision. We add six categories transform method, `resize`, `random crop`,`center crop`, `random horizontal flip`, `to tensor`. `normaliza`.

- **Resize**: Resize the input image to the given size. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
- **Random crop**: Crop the given image at a random location. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions, but if non-constant padding is used, the input is expected to have at most 2 leading dimensions
- **Center crop**: Crops the given image at the center. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
- **Random horizontal flip**: Horizontally flip the given image randomly with a given probability. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
- **To tensor**: Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
- **Normalize**: Normalize a tensor image with mean and standard deviation.

For more details about transforms, see [torchvision.transforms](https://pytorch.org/vision/0.11/transforms.html)

We import the `torchvision` and `PIL` to define the transform functions.

And in "fgvclib/transforms/__init__.py", we define the function `get_transform` to return the transforms with the given name. The given names are `resize`, `center_crop`, `random_crop`, `random_horizontal_flip`, `to_tensor`, `normalize`.
```python
def get_transform(transform_name):
    """Return the backbone with the given name."""
    if transform_name not in globals():
        raise NotImplementedError(f"Transform not found: {transform_name}\nAvailable transforms: {__all__}")
    return globals()[transform_name]
```

## The example
The parameters about the network are saved in the configs in advance, so we can build transforms for train or test dataset according to config.
```python
from fgvclib.transforms import get_transform

def build_transforms(transforms_cfg: CfgNode) -> transforms.Compose:
    """
    Args:
        transforms_cfg (CfgNode): The root config node.
    Returns:
        transforms.Compose: The transforms.Compose object in Pytorch.
    """
    return transforms.Compose([get_transform(item['name'])(item) for item in transforms_cfg])
```