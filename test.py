import torch
from PIL import Image
import torchvision
import torchvision.transforms.v2 as T

print(f"正在使用 TorchVision 版本: {torchvision.__version__}")

# 1. 创建一个虚拟的 PIL Image, 模拟从数据集中加载的图片
try:
    dummy_pil_image = Image.new('RGB', (256, 256), color='red')
    print(f"输入类型: {type(dummy_pil_image)}")
except Exception as e:
    print(f"创建虚拟图片失败，请确保 Pillow 已安装: {e}")
    exit()

# 2. 定义一个完整的、一体化的处理流程
#    这个流程包含了之前报错的所有关键步骤
transform_pipeline = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2),  # 在 PIL Image 上操作
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),         # 关键的类型转换步骤
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 在 Tensor 上操作
])

print("\n正在尝试应用变换...")

# 3. 运行测试并捕获可能出现的错误
try:
    # 应用完整的处理流程
    output_tensor = transform_pipeline(dummy_pil_image)
    
    # 4. 检查输出结果
    print("\n✅ 验证成功！问题已解决。")
    print(f"输出类型: {type(output_tensor)}")
    print(f"输出 Tensor 的形状: {output_tensor.shape}")
    
except TypeError as e:
    print("\n❌ 验证失败！问题仍然存在。")
    print(f"捕获到的错误: {e}")
except Exception as e:
    print(f"\n❌ 验证失败！发生了意外的错误: {e}")