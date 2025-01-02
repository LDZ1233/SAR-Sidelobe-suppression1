import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import DenoisingDataset
from net import ChangeDetectionNetwork
from tqdm import tqdm
import torch.nn.functional as F


class ContentLoss(nn.Module):
    """内容损失：结合MSE和感知损失"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # 基础MSE损失
        mse_loss = self.mse(output, target)

        # 梯度损失 - 保持边缘清晰
        def gradient_loss(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy

        output_dx, output_dy = gradient_loss(output)
        target_dx, target_dy = gradient_loss(target)

        gradient_loss = F.l1_loss(output_dx, target_dx) + F.l1_loss(output_dy, target_dy)

        # 结构相似性损失
        ssim_loss = 1 - torch.mean(structural_similarity(output, target))

        # 总损失
        total_loss = mse_loss + 0.5 * gradient_loss + 0.5 * ssim_loss

        return total_loss


def structural_similarity(img1, img2, window_size=11, sigma=1.5):
    """计算结构相似性"""
    channel = img1.size(1)

    # 创建高斯核
    kernel_size = window_size
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2
    g = coords ** 2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channel, 1, kernel_size, kernel_size).to(img1.device)

    # 计算均值和方差
    mu1 = F.conv2d(img1, kernel, padding=kernel_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=kernel_size // 2, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size // 2, groups=channel) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def train_denoising_network(
        dataset_path,
        batch_size=4,
        lr=2e-4,  # 降低初始学习率
        num_epochs=50,  # 增加训练轮数
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Using device: {device}")

    # 数据预处理
    transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    # 数据集和数据加载器
    dataset = DenoisingDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # 初始化模型
    model = ChangeDetectionNetwork(input_channels=1).to(device)

    # 使用改进的损失函数
    criterion = ContentLoss().to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 记录最佳模型
    best_loss = float('inf')

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in loop:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            target = batch["target"].to(device)

            # 前向传播
            output = model(x1, x2)

            # 计算损失
            loss = criterion(output, target)
            epoch_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 更新进度条
            loop.set_postfix(loss=loss.item())

        # 更新学习率
        scheduler.step()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_denoising_model.pth")
            print(f"Saved best model with loss: {best_loss:.4f}")

    print("Training completed!")


if __name__ == "__main__":
    dataset_path = "output_images"

    # 训练
    train_denoising_network(
        dataset_path,
        batch_size=16,
        num_epochs=50
    )