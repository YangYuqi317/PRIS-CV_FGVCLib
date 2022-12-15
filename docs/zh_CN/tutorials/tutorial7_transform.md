# Tutorial 7: Learn about transforms

我们引入`transforms`来处理图片，我们导入了六类转换的方法`resize`, `random crop`,`center crop`, `random horizontal flip`, `to tensor`. `normaliza`

- **Resize**: 将图像调整为给定的大小. 
- **Random crop**: 在随机位置裁剪给定的图像. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions, but if non-constant padding is used, the input is expected to have at most 2 leading dimensions
- **Center crop**:在中心裁剪给定的图像 If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
- **Random horizontal flip**: 以给定的概率随机地水平翻转给定的图像. If the image is torch Tensor, it is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions
- **To tensor**: 转换PIL Image或numpy.ndarry到tensor
- **Normalize**: 用均值和标准超归一化图像

关于转换的更多细节请参见[torchvision.transforms](https://pytorch.org/vision/0.11/transforms.html)

我们导入了`torchvision` 和`PIL`去定义转换函数

在"fgvclib/transforms/__init__.py"中，我们定义了函数`get_transform`，根据给定的转换类型返回转换函数，给定的转换类型有：`resize`, `center_crop`, `random_crop`, `random_horizontal_flip`, `to_tensor`, `normalize`

```python
def get_transform(transform_name):
    """Return the backbone with the given name."""
    if transform_name not in globals():
        raise NotImplementedError(f"Transform not found: {transform_name}\nAvailable transforms: {__all__}")
    return globals()[transform_name]
```

## 举例
网络参数事先保存在配置中，可以根据配置对训练数据集或测试数据集进行变换。

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