import os, time, copy, gc, glob, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from timm.models.layers import drop_path
from thop import profile
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import numpy as np

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
DATASET_NAME = "Oxford_IIIT_Pet"
NUM_CLASSES = 37 # Oxford-IIIT Pet has 37 categories

# Base directories based on terminal structure
PARENT_DIR = "/tmp/rohan_workspace/AGMBiN/"
DATA_DIR = os.path.join(PARENT_DIR, "data/Oxford_IIIT_Pet/images")
PRETRAINED_WEIGHTS = os.path.join(PARENT_DIR, "models/model_AGMBiN_IN1K_150e.pth")

# Output files routing
FINETUNE_DIR = os.path.join(PARENT_DIR, "finetune")
os.makedirs(FINETUNE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(FINETUNE_DIR, f"ft_model_AGMBiN_{DATASET_NAME}.pth")
LOG_FILE_PATH = os.path.join(FINETUNE_DIR, f"ft_training_log_AGMBiN_{DATASET_NAME}.txt")
METRICS_FILE_PATH = os.path.join(FINETUNE_DIR, f"ft_eval_metrics_AGMBiN_{DATASET_NAME}.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

# =============================================================================
# 2. DATA HANDLING (CUSTOM FILENAME PARSING)
# =============================================================================
class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Grab only .jpg images (ignoring the .mat files shown in the terminal)
        all_images = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        
        # Parse class names from filenames
        dataset_items = []
        classes = set()
        
        for img_path in all_images:
            filename = os.path.basename(img_path)
            class_name = "_".join(filename.split("_")[:-1])
            if class_name: # Handle any weird filenames
                classes.add(class_name)
                dataset_items.append({'path': img_path, 'class_name': class_name})
            
        self.classes = sorted(list(classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Group items by class for stratified splitting
        class_groups = defaultdict(list)
        for item in dataset_items:
            class_groups[item['class_name']].append({
                'path': item['path'],
                'label': self.class_to_idx[item['class_name']]
            })
            
        # Create deterministic 80/20 train/val split
        train_items = []
        val_items = []
        
        for class_name, items in class_groups.items():
            random.Random(42).shuffle(items)
            split_idx = int(0.8 * len(items))
            train_items.extend(items[:split_idx])
            val_items.extend(items[split_idx:])
            
        self.data = train_items if is_train else val_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['label']

# UPDATED DATA PIPELINE (448x448 + Strong Augmentation)
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)) 
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = OxfordPetDataset(DATA_DIR, is_train=True, transform=transform_train)
val_ds = OxfordPetDataset(DATA_DIR, is_train=False, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# =============================================================================
# 3. ARCHITECTURE (AGMBiN)
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
    def __init__(self, num_classes=10, base_channels=64, bilinear_dim=128):
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
        self.attn_pool = nn.Sequential(nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=1), nn.Sigmoid())
        self.compress = nn.Sequential(nn.Conv2d(base_channels * 16, bilinear_dim, kernel_size=1, bias=False), nn.BatchNorm2d(bilinear_dim), nn.GELU())
        self.fc = nn.Linear(bilinear_dim * bilinear_dim, num_classes)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(x))))))
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
        return logits

# =============================================================================
# 4. COMPREHENSIVE EVALUATION METRICS
# =============================================================================
def evaluate_comprehensive_metrics(model, dataloader, device):
    """Calculates all advanced metrics and handles inference timing."""
    model.eval()
    all_preds = []
    all_targets = []
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                
            # Top-1 and Top-5 logic
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            
            correct_top1 += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += targets.size(0)
            
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    end_time = time.time()
    
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    
    # MCA, Precision, Recall, F1
    mean_class_acc = balanced_accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    
    # Timing Metrics
    total_time = end_time - start_time
    throughput = total / total_time 
    latency = (total_time / total) * 1000 
    
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
    return {
        "Top-1 Acc": top1_acc,
        "Top-5 Acc": top5_acc,
        "Mean Class Acc": mean_class_acc,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1-Score": f1,
        "Throughput (img/s)": throughput,
        "Latency (ms/img)": latency,
        "Peak VRAM (MB)": peak_vram_mb
    }

# =============================================================================
# 5. TRAINING SETUP & LOGGING
# =============================================================================
def log_to_file(message, filepath=LOG_FILE_PATH):
    print(message)
    with open(filepath, "a") as f:
        f.write(message + "\n")

model = AGMBiN(num_classes=NUM_CLASSES).to(device)

if os.path.exists(PRETRAINED_WEIGHTS):
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    model.load_state_dict(state_dict, strict=False)
    log_to_file(f"✅ Pretrained weights loaded for {DATASET_NAME}")
else:
    log_to_file(f"⚠️ Pretrained weights NOT FOUND at {PRETRAINED_WEIGHTS}. Training from scratch.")

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES).to(device)

# ---------------------------------------------------------
# Calculate FLOPs and Parameters
# ---------------------------------------------------------
dummy_input = torch.randn(1, 3, 448, 448).to(device)
macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
flops_g = (macs * 2) / 1e9  
params_m = params / 1e6     

log_to_file("========================================")
log_to_file(f"📊 Dataset: {DATASET_NAME} ({NUM_CLASSES} classes)")
log_to_file(f"⚙️  Model Params: {params_m:.2f} M")
log_to_file(f"🚀 Model FLOPs:  {flops_g:.2f} G")
log_to_file("========================================")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

def get_param_groups(lr_backbone, lr_head):
    backbone, head = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            head.append(param)
        else:
            backbone.append(param)
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head}
    ]

def set_trainable(stage):
    for name, param in model.named_parameters():
        if stage == "warmup":
            param.requires_grad = 'fc' in name
        elif stage == "full":
            param.requires_grad = True

# =============================================================================
# 6. TRAIN LOOP
# =============================================================================
def train_phase(epochs, optimizer, scheduler, stage, phase_name):
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    set_trainable(stage)
    
    # Updated to include Time(m:s) in the header
    log_to_file(f"Epoch\tTrain_Loss\tVal_Acc\t\tLR\t\tTime(m:s)\tStatus")

    for epoch in range(epochs):
        epoch_start_time = time.time() # Capture start time of the epoch
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # Fast Validation for training loop tracking
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                pred = out.argmax(1)
                total += y.size(0)
                correct += (pred == y).sum().item()

        acc = correct / total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, MODEL_SAVE_PATH)
            flag = "💾 BEST"
        else:
            flag = ""
            
        epoch_end_time = time.time() # Capture end time of the epoch
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_mins = int(epoch_duration // 60)
        epoch_secs = int(epoch_duration % 60)
        time_str = f"{epoch_mins:02d}:{epoch_secs:02d}"

        # Log details with the calculated time
        log_to_file(f"{epoch+1:02d}/{epochs}\t{train_loss/len(train_loader):.4f}\t\t{acc*100:.2f}%\t\t{current_lr:.6f}\t{time_str}\t\t{flag}")

    return best_wts

# =============================================================================
# 7. EXECUTION
# =============================================================================
if __name__ == "__main__":
    
    total_start_time = time.time() # Record the start of the entire execution
    
    # -------- Phase 1: Aggressive Head Warmup (5 Epochs) --------
    log_to_file("\n🔹 Phase 1: Warmup (Linear)")
    opt1 = optim.AdamW(get_param_groups(0.0, 1e-3), weight_decay=5e-4)
    sch1 = optim.lr_scheduler.LinearLR(opt1, start_factor=0.1, total_iters=5)
    
    wts = train_phase(5, opt1, sch1, "warmup", "Warmup")
    model.load_state_dict(wts)

    # -------- Phase 2: Deep Joint Fine-Tuning (50 Epochs) --------
    log_to_file("\n🔹 Phase 2: Full Network Joint Fine-Tuning")
    opt2 = optim.AdamW(get_param_groups(5e-5, 5e-4), weight_decay=5e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50, eta_min=1e-6)
    
    wts = train_phase(50, opt2, sch2, "full", "Joint FT")
    model.load_state_dict(wts)
    
    log_to_file(f"\n✅ Training Completed. Model saved to {MODEL_SAVE_PATH}")
    
    # -------- FINAL EVALUATION METRICS --------
    log_to_file("\n📊 Running Final Comprehensive Evaluation on Validation Set...")
    metrics_results = evaluate_comprehensive_metrics(model, val_loader, device)
    
    log_to_file("\n================ FINAL METRICS ================")
    for k, v in metrics_results.items():
        if "Acc" in k or "Precision" in k or "Recall" in k or "F1" in k:
            log_to_file(f"{k}: {v*100:.2f}%")
        else:
            log_to_file(f"{k}: {v:.2f}")
    log_to_file(f"Model Params: {params_m:.2f} M")
    log_to_file(f"Model FLOPs: {flops_g:.2f} G")
    log_to_file("===============================================")
    
    # Save to dedicated metrics file
    with open(METRICS_FILE_PATH, "w") as f:
        f.write(f"=== Metrics for {DATASET_NAME} ===\n")
        f.write(f"Model Params: {params_m:.2f} M\n")
        f.write(f"Model FLOPs: {flops_g:.2f} G\n")
        for k, v in metrics_results.items():
             f.write(f"{k}: {v}\n")
    log_to_file(f"Metrics saved to {METRICS_FILE_PATH}")
    
    total_end_time = time.time() # Record the end of the entire execution
    total_duration = total_end_time - total_start_time
    total_hours = int(total_duration // 3600)
    total_mins = int((total_duration % 3600) // 60)
    total_secs = int(total_duration % 60)
    
    # Log total execution time
    log_to_file(f"\n⏱️ Total Execution Time: {total_hours:02d}h {total_mins:02d}m {total_secs:02d}s")
    
    del model, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()