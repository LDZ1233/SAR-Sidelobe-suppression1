import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DenoisingDataset(Dataset):
    """
    自定义去噪数据集类
    """

    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据根目录，包含 "filtered", "min", "original" 子目录
        :param transform: 图像预处理操作
        """
        self.root_dir = root_dir
        self.transform = transform

        # 获取每个目录中的文件路径
        self.filtered_dir = os.path.join(root_dir, "filtered")  # x2
        self.min_dir = os.path.join(root_dir, "min")  # 目标值
        self.original_dir = os.path.join(root_dir, "original")  # x1

        self.filtered_files = sorted(os.listdir(self.filtered_dir))
        self.min_files = sorted(os.listdir(self.min_dir))
        self.original_files = sorted(os.listdir(self.original_dir))

        # 确保文件数量一致
        assert len(self.filtered_files) == len(self.min_files) == len(self.original_files), \
            "三个目录中的文件数量不一致，请检查数据完整性。"

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.filtered_files)

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        """
        # 加载图像
        filtered_path = os.path.join(self.filtered_dir, self.filtered_files[idx])
        min_path = os.path.join(self.min_dir, self.min_files[idx])
        original_path = os.path.join(self.original_dir, self.original_files[idx])

        filtered_image = Image.open(filtered_path).convert("L")  # x2
        min_image = Image.open(min_path).convert("L")  # 目标值
        original_image = Image.open(original_path).convert("L")  # x1

        # 应用图像预处理
        if self.transform:
            filtered_image = self.transform(filtered_image)
            min_image = self.transform(min_image)
            original_image = self.transform(original_image)

        # 返回字典
        return {
            "x2": filtered_image,  # 输入 x2
            "target": min_image,  # 目标值
            "x1": original_image  # 输入 x1
        }


# 测试数据集
if __name__ == "__main__":
    # 数据路径
    root_dir = "output_images"

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor()  # 转换为张量
    ])

    # 实例化数据集
    dataset = DenoisingDataset(root_dir, transform=transform)

    # 测试访问
    sample = dataset[0]
    print(f"x2 shape: {sample['x2'].shape}")
    print(f"target shape: {sample['target'].shape}")
    print(f"x1 shape: {sample['x1'].shape}")
