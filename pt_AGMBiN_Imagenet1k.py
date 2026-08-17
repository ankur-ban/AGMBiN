# -*- coding: utf-8 -*-
import os
import time
import math
import subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# =============================================================================
# 0. CONFIGURATION & WORKSPACE SETUP
# =============================================================================

DEBUG_MODE = False

# We are only keeping pre-training on ImageNet-1K
PRETRAIN_EPOCHS = 150 if not DEBUG_MODE else 1

# Batch size scaled up for A40 (48GB VRAM)
BATCH_SIZE = 256 if not DEBUG_MODE else 32
NUM_WORKERS = 8 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Centralized Workspace on the High-Capacity Drive
BASE_WORKSPACE = '/tmp/rohan_workspace'
DATA_DIR = os.path.join(BASE_WORKSPACE, 'data')

DIRS = [
    os.path.join(BASE_WORKSPACE, 'ag_mc_workspace/models'),
    os.path.join(BASE_WORKSPACE, 'ag_mc_workspace/csv_logs'),
    DATA_DIR
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

# Standard ImageNet Transforms (Resolution 224x224)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# =============================================================================
# 1. Helper Functions & Losses
# =============================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

def get_mixup_cutmix(imgs, lbls, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = imgs.size(0)
    index = torch.randperm(batch_size).to(imgs.device)
    mixed_imgs = lam * imgs + (1 - lam) * imgs[index, :]
    lbls_a, lbls_b = lbls, lbls[index]
    return mixed_imgs, lbls_a, lbls_b, lam

def manifold_consistency_loss(manifold_features, temp=0.5):
    total_loss = 0.0
    start_idx = max(1, len(manifold_features) - 3)
    count = 0
    for i in range(start_idx, len(manifold_features)):
        f_shallow = manifold_features[i-1].detach() 
        f_deep = manifold_features[i]
        sim_shallow = F.softmax(torch.mm(f_shallow, f_shallow.t()) / (temp * math.sqrt(128)), dim=1)
        sim_deep = F.softmax(torch.mm(f_deep, f_deep.t()) / (temp * math.sqrt(128)), dim=1)
        total_loss += F.mse_loss(sim_deep, sim_shallow)
        count += 1
    return total_loss / count if count > 0 else torch.tensor(0.0).to(manifold_features[0].device)

# =============================================================================
# 2. Architecture: AG-MC-BCNN
# =============================================================================
class ModernManifoldBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_path_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop_path_rate = drop_path_rate

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(1, out_channels // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(1, out_channels // 4), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(self.act(self.conv1(x)))
        out = self.bn2(self.act(self.conv2(out)))
        out = out * self.se(out)
        out = drop_path(out, self.drop_path_rate, self.training)
        out += identity
        return self.act(out)

class AGMBiN(nn.Module):
    def __init__(self, num_classes=1000, base_channels=64, bilinear_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )
        self.layer1 = ModernManifoldBlock(base_channels, base_channels * 2, stride=2)
        self.layer2 = ModernManifoldBlock(base_channels * 2, base_channels * 4, stride=2)
        self.layer3 = ModernManifoldBlock(base_channels * 4, base_channels * 8, stride=2)
        self.layer4 = ModernManifoldBlock(base_channels * 8, base_channels * 16, stride=1)
        self.layer5 = ModernManifoldBlock(base_channels * 16, base_channels * 16, stride=2)

        self.attn_pool = nn.Sequential(
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=1),
            nn.Sigmoid()
        )
        self.compress = nn.Sequential(
            nn.Conv2d(base_channels * 16, bilinear_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bilinear_dim),
            nn.GELU()
        )
        # Output directly to 1000 classes for ImageNet
        self.fc = nn.Linear(bilinear_dim * bilinear_dim, num_classes)

        self.global_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(c, 128), nn.LayerNorm(128))
            for c in [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16, base_channels*16]
        ])
        self._initialize_weights()

    def forward(self, x):
        manifold_features = []
        x = self.conv1(x)
        manifold_features.append(self.projectors[0](self.global_avg(x).view(x.size(0), -1)))

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]):
            x = layer(x)
            manifold_features.append(self.projectors[i+1](self.global_avg(x).view(x.size(0), -1)))

        x = x * self.attn_pool(x)
        x = self.compress(x)
        B, C, H, W = x.size()

        with torch.amp.autocast('cuda', enabled=False):
            x_flat = x.float().view(B, C, H * W)
            x_flat = torch.clamp(x_flat, min=-100.0, max=100.0) 
            bilinear = torch.bmm(x_flat, x_flat.transpose(1, 2)) / (H * W)
            bilinear = bilinear.view(B, C * C)
            bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-12)
            bilinear = F.normalize(bilinear, p=2, dim=1)
            logits = self.fc(bilinear)

        return logits, manifold_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# =============================================================================
# 3. DATASET LOADERS (IMAGENET-1K ONLY)
# =============================================================================

def get_imagenet_data():
    print("🌍 Setting up Full ImageNet-1K Dataset...")
    imagenet_dir = os.path.join(DATA_DIR, 'imagenet1k')
    train_dir = os.path.join(imagenet_dir, 'ILSVRC', 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(imagenet_dir, 'ILSVRC', 'Data', 'CLS-LOC', 'val')

    if not os.path.exists(train_dir):
        print("⚠️ ImageNet not found! You must download it via Kaggle API:")
        print(f"kaggle competitions download -c imagenet-object-localization-challenge -p {DATA_DIR}")
        print(f"unzip -q {DATA_DIR}/imagenet-object-localization-challenge.zip -d {imagenet_dir}")
        raise FileNotFoundError("ImageNet dataset is missing. Please download it first.")

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)

    if DEBUG_MODE:
        train_ds = Subset(train_ds, range(100))
        val_ds = Subset(val_ds, range(100))

    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valloader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return trainloader, valloader, 1000

# =============================================================================
# 4. TRAINING & EVALUATION ENGINE
# =============================================================================

def train_epoch(model, dataloader, optimizer, criterion, apply_mixup=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    total_batches = len(dataloader)
    pbar = tqdm(dataloader, desc="Training", leave=False, colour='blue')

    for i, (inputs, targets) in enumerate(pbar):
        if i >= total_batches / 2:
            pbar.colour = 'green'

        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        if apply_mixup and np.random.rand() < 0.5:
            mixed_inputs, targets_a, targets_b, lam = get_mixup_cutmix(inputs, targets)
            logits, manifold_features = model(mixed_inputs)
            ce_loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
        else:
            logits, manifold_features = model(inputs)
            ce_loss = criterion(logits, targets)

        mc_loss = manifold_consistency_loss(manifold_features)
        loss = ce_loss + 0.1 * mc_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)

        if apply_mixup and 'targets_a' in locals():
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return running_loss / len(dataloader), 100. * correct / total

def eval_epoch(model, dataloader, criterion, return_predictions=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False, colour='magenta')

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

    acc = 100. * correct / total
    if return_predictions:
        return running_loss / len(dataloader), acc, all_preds, all_targets
    return running_loss / len(dataloader), acc

def run_training_pipeline(model, model_name, trainloader, valloader, epochs, phase_name, optimizer, scheduler=None, save_path=None, apply_mixup=True):
    criterion = nn.CrossEntropyLoss()
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    print(f"\n--- Starting {phase_name} for {model_name} ({epochs} Epochs) ---")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, apply_mixup)
        val_loss, val_acc = eval_epoch(model, valloader, criterion)

        if scheduler: scheduler.step()

        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_acc and save_path:
            best_acc = val_acc
            # Save checkpoint (handle DataParallel wrapping safely)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"💾 Checkpoint saved! New best accuracy: {best_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"⏱️ Total {phase_name} time for {model_name}: {total_time / 60:.2f} minutes.")
    return history

# =============================================================================
# 5. MASTER EXECUTION SCRIPT
# =============================================================================

def get_multi_gpu_model(num_classes):
    """Initializes model and wraps in DataParallel if multiple GPUs are detected."""
    model = AGMBiN(num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"🔥 Utilizing {torch.cuda.device_count()} GPUs via DataParallel!")
        model = nn.DataParallel(model)
    return model.to(DEVICE)

def main():
    print(f"🚀 Starting AG-MC-BCNN ImageNet-1K Pre-Training Pipeline on {DEVICE}")
    model_name = 'AGMBiN_ImageNet1K'

    print("\n" + "="*50 + f"\nPRE-TRAINING ({model_name} on ImageNet-1K)\n" + "="*50)
    pt_trainloader, pt_valloader, pt_num_classes = get_imagenet_data()
    pt_save_path = os.path.join(BASE_WORKSPACE, f'ag_mc_workspace/models/pretrained_{model_name}.pth')

    if os.path.exists(pt_save_path) and not DEBUG_MODE:
        print(f"✅ Found existing pre-trained weights at {pt_save_path}.")
        print("Training is already complete. Exiting.")
        return

    model = get_multi_gpu_model(pt_num_classes)
    
    # Optimizer and Scheduler for 150 epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETRAIN_EPOCHS)

    hist = run_training_pipeline(
        model, model_name, pt_trainloader, pt_valloader,
        epochs=PRETRAIN_EPOCHS, phase_name="ImageNet Pre-Training",
        optimizer=optimizer, scheduler=scheduler, save_path=pt_save_path, 
        apply_mixup=True # Highly recommended for 150 epochs on 1.2M images
    )
    
    # Save training logs
    csv_path = os.path.join(BASE_WORKSPACE, f'ag_mc_workspace/csv_logs/pretrain_{model_name}.csv')
    pd.DataFrame(hist).to_csv(csv_path, index=False)
    
    print("\n🎉 ImageNet-1K Pre-Training COMPLETE!")
    print(f"The final weights are saved at: {pt_save_path}")
    print(f"The training logs are saved at: {csv_path}")

if __name__ == '__main__':
    main()