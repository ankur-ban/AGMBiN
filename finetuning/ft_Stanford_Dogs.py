import os, time, copy, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import splitfolders
from timm.models.layers import drop_path
from thop import profile
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
DATASET_NAME = "Stanford_Dogs"
NUM_CLASSES = 120

# Base directories based on terminal structure
PARENT_DIR = "/tmp/rohan_workspace/AGMBiN/"
RAW_DATA_DIR = os.path.join(PARENT_DIR, "data/stanford_dogs/")
SPLIT_DATA_DIR = os.path.join(PARENT_DIR, "data/stanford_dogs_split")
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
# 2. DATA HANDLING & AUTOMATIC SPLITTING
# =============================================================================
# Automatically create train/val split folders if not already present
if not os.path.exists(os.path.join(SPLIT_DATA_DIR, 'train')):
    print(f"🔀 Splitting dataset from {RAW_DATA_DIR} into {SPLIT_DATA_DIR} (80% train / 20% val)...")
    splitfolders.ratio(RAW_DATA_DIR, output=SPLIT_DATA_DIR, seed=42, ratio=(0.8, 0.2))

# UPDATED DATA PIPELINE (448x448 + Strong Augmentation)
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9), # Strong structural variation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)) # Forces multi-context attention
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(SPLIT_DATA_DIR, 'train'), transform=transform_train)
val_ds = datasets.ImageFolder(os.path.join(SPLIT_DATA_DIR, 'val'), transform=transform_val)

# Reduced batch size to 16 to avoid Out-Of-Memory (OOM) errors on A40 GPU at 448x448
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
# 4. TRAINING SETUP & LOGGING
# =============================================================================
def log_to_file(message, path=LOG_FILE_PATH):
    print(message)
    with open(path, "a") as f:
        f.write(message + "\n")

model = AGMBiN(num_classes=NUM_CLASSES).to(device)

if os.path.exists(PRETRAINED_WEIGHTS):
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    model.load_state_dict(state_dict, strict=False)
    log_to_file(f"✅ Pretrained weights loaded for {DATASET_NAME} from {PRETRAINED_WEIGHTS}")
else:
    log_to_file(f"⚠️ Pretrained weights NOT FOUND at {PRETRAINED_WEIGHTS}. Training from scratch.")

model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.amp.GradScaler('cuda')

# ===================== PARAM GROUPS & FREEZE STRATEGY =====================
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
        elif stage == "partial":
            param.requires_grad = ('layer4' in name) or ('layer5' in name) or ('fc' in name)
        elif stage == "full":
            param.requires_grad = True

# ===================== 3-PHASE TRAIN LOOP =====================
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

        # Validation
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
# 5. ADVANCED METRICS EVALUATION & FILE SAVING
# =============================================================================
def evaluate_advanced_metrics():
    log_to_file(f"\n{'='*50}", path=METRICS_FILE_PATH)
    log_to_file(f"📊 ADVANCED EVALUATION METRICS REPORT: {DATASET_NAME}", path=METRICS_FILE_PATH)
    log_to_file(f"{'='*50}", path=METRICS_FILE_PATH)
    
    # Load best saved model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    all_targets = []
    all_preds_top1 = []
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(x)
            
            # Calculate Top-1 and Top-5 Accuracy
            _, pred = logits.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.view(1, -1).expand_as(pred))
            
            correct_top1 += correct[0].reshape(-1).float().sum(0, keepdim=True).item()
            correct_top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += y.size(0)
            
            all_targets.extend(y.cpu().numpy())
            all_preds_top1.extend(pred[0].cpu().numpy())

    # Accuracy Metrics
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    
    # Sklearn Metrics
    mca = balanced_accuracy_score(all_targets, all_preds_top1)
    macro_prec = precision_score(all_targets, all_preds_top1, average='macro', zero_division=0)
    macro_rec = recall_score(all_targets, all_preds_top1, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds_top1, average='macro', zero_division=0)
    
    # Latency & Throughput Benchmark
    dummy_input = torch.randn(1, 3, 448, 448).to(device)
    
    for _ in range(10): # GPU Warmup
        model(dummy_input)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        model(dummy_input)
    torch.cuda.synchronize()
    end_time = time.time()
    
    latency_ms = ((end_time - start_time) / 100) * 1000
    throughput = 1000 / latency_ms
    
    # FLOPs and Parameters Calculation
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops_g = (macs * 2) / 1e9
    params_m = params / 1e6
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) # in GB

    # Save to distinct metrics file
    log_to_file(f"1. Top-1 Accuracy:        {top1_acc*100:.2f}%", path=METRICS_FILE_PATH)
    log_to_file(f"2. Top-5 Accuracy:        {top5_acc*100:.2f}%", path=METRICS_FILE_PATH)
    log_to_file(f"3. Mean Class Acc (MCA):  {mca*100:.2f}%", path=METRICS_FILE_PATH)
    log_to_file(f"4. Macro Precision:       {macro_prec:.4f}", path=METRICS_FILE_PATH)
    log_to_file(f"5. Macro Recall:          {macro_rec:.4f}", path=METRICS_FILE_PATH)
    log_to_file(f"6. Macro F1-Score:        {macro_f1:.4f}", path=METRICS_FILE_PATH)
    log_to_file(f"----------------------------------------", path=METRICS_FILE_PATH)
    log_to_file(f"7. Inference Latency:     {latency_ms:.2f} ms/image", path=METRICS_FILE_PATH)
    log_to_file(f"8. Inference Throughput:  {throughput:.2f} images/sec", path=METRICS_FILE_PATH)
    log_to_file(f"9. Peak VRAM Usage:       {peak_vram:.2f} GB", path=METRICS_FILE_PATH)
    log_to_file(f"10. Model Parameters:     {params_m:.2f} M", path=METRICS_FILE_PATH)
    log_to_file(f"11. Model FLOPs:          {flops_g:.2f} G", path=METRICS_FILE_PATH)
    log_to_file(f"{'='*50}\n", path=METRICS_FILE_PATH)


# =============================================================================
# 🔥 EXECUTION: TRAINING PHASES WITH COSINE ANNEALING
# =============================================================================
if __name__ == "__main__":
    total_start_time = time.time() # Record the start of the entire execution
    
    # -------- Phase 1: Aggressive Head Warmup (5 Epochs) --------
    log_to_file("\n🔹 Phase 1: Warmup (Linear)")
    # High learning rate for the initialized FC layer
    opt1 = optim.AdamW(get_param_groups(0.0, 1e-3), weight_decay=5e-4)
    # Using a simple Linear Warmup to rapidly stabilize the head
    sch1 = optim.lr_scheduler.LinearLR(opt1, start_factor=0.1, total_iters=5)
    
    wts = train_phase(5, opt1, sch1, "warmup", "Warmup")
    model.load_state_dict(wts)

    # -------- Phase 2: Deep Joint Fine-Tuning (50 Epochs) --------
    log_to_file("\n🔹 Phase 2: Full Network Joint Fine-Tuning")
    # Differential LR: Backbone learns 10x slower than the head
    opt2 = optim.AdamW(get_param_groups(5e-5, 5e-4), weight_decay=5e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50, eta_min=1e-6)
    
    wts = train_phase(50, opt2, sch2, "full", "Joint FT")
    model.load_state_dict(wts)
    
    log_to_file(f"\n✅ Training Completed for {DATASET_NAME}. Running advanced metric evaluation...")
    
    # Run evaluation and save to evaluation_metrics_AGMBiN_Food_101.txt
    evaluate_advanced_metrics()
    print(f"📁 Advanced metrics saved to {METRICS_FILE_PATH}")
    
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