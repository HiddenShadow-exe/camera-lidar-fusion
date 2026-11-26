import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from shapely.geometry import Polygon

# ==============================================================================
# 1. Dataset & DataLoader
# ==============================================================================
class OBBDataset(Dataset):
    def __init__(self, root_dir, split="train", num_points=2048):
        self.num_points = num_points
        self.paths = list(Path(root_dir).glob(f"{split}/*/*/recording_data.json"))
        self.samples = []
        
        # Pre-load and parse all JSONs into RAM once
        for path in self.paths:
            with open(path, 'r') as f:
                data = json.load(f)
            points = np.array(data["3d_data_camera"]["object_points"], dtype=np.float32)
            obb = data["obb_label"]
            center = np.array(obb["center"], dtype=np.float32)
            dimensions = np.array(obb["dimensions"], dtype=np.float32)
            R = np.array(obb["rotation"], dtype=np.float32).reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            yaw_cs = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            self.samples.append((points, center, dimensions, yaw_cs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_points, center, dimensions, target_yaw_cos_sin = self.samples[idx]
        num_raw = len(raw_points)
        
        if num_raw == 0:
            points = np.zeros((self.num_points, 3), dtype=np.float32)
        elif num_raw >= self.num_points:
            choices = np.random.choice(num_raw, self.num_points, replace=False)
            points = raw_points[choices].copy()
        else:
            choices = np.random.choice(num_raw, self.num_points, replace=True)
            points = raw_points[choices].copy()
            
        centroid = np.mean(points, axis=0)
        points -= centroid
        target_center = center - centroid
        
        return {
            "points": torch.from_numpy(points),
            "center": torch.from_numpy(target_center),
            "dimensions": torch.from_numpy(dimensions),
            "yaw_cos_sin": torch.from_numpy(target_yaw_cos_sin),
            "centroid": torch.from_numpy(centroid)
        }

# ==============================================================================
# 2. PointNet Model
# ==============================================================================
class PointNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class ClusterToOBBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PointNetBackbone()
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.3)
        
        self.head_center = nn.Linear(256, 3)
        self.head_dim = nn.Linear(256, 3)
        self.head_yaw = nn.Linear(256, 2)

    def forward(self, x):
        if x.shape[-1] == 3:
            x = x.transpose(1, 2)
            
        global_feat = self.backbone(x)
        
        feat = F.relu(self.bn1(self.fc1(global_feat)))
        feat = self.drop(feat)
        feat = F.relu(self.bn2(self.fc2(feat)))
        
        center = self.head_center(feat) 
        dimensions = F.softplus(self.head_dim(feat))
        yaw_normalized = F.normalize(self.head_yaw(feat), p=2, dim=1)
        
        return {"center": center, "dimensions": dimensions, "yaw_cos_sin": yaw_normalized}

# ==============================================================================
# 3. 3D Evaluation Metric
# ==============================================================================
def get_bev_corners(cx, cy, l, w, yaw):
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    dx1, dy1 = l/2 * cos_y, l/2 * sin_y
    dx2, dy2 = w/2 * -sin_y, w/2 * cos_y
    
    return [
        (cx + dx1 + dx2, cy + dy1 + dy2),
        (cx + dx1 - dx2, cy + dy1 - dy2),
        (cx - dx1 - dx2, cy - dy1 - dy2),
        (cx - dx1 + dx2, cy - dy1 + dy2)
    ]

def compute_3d_iou(pred_c, pred_d, pred_yaw, tgt_c, tgt_d, tgt_yaw):
    pred_z_min, pred_z_max = pred_c[2] - pred_d[2]/2, pred_c[2] + pred_d[2]/2
    tgt_z_min, tgt_z_max = tgt_c[2] - tgt_d[2]/2, tgt_c[2] + tgt_d[2]/2
    
    inter_z_min = max(pred_z_min, tgt_z_min)
    inter_z_max = min(pred_z_max, tgt_z_max)
    inter_h = max(0, inter_z_max - inter_z_min)
    
    if inter_h == 0:
        return 0.0
        
    pred_poly = Polygon(get_bev_corners(pred_c[0], pred_c[1], pred_d[0], pred_d[1], pred_yaw))
    tgt_poly = Polygon(get_bev_corners(tgt_c[0], tgt_c[1], tgt_d[0], tgt_d[1], tgt_yaw))
    
    if not pred_poly.intersects(tgt_poly):
        return 0.0
        
    inter_area = pred_poly.intersection(tgt_poly).area
    inter_vol = inter_area * inter_h
    pred_vol = pred_d[0] * pred_d[1] * pred_d[2]
    tgt_vol = tgt_d[0] * tgt_d[1] * tgt_d[2]
    
    union_vol = pred_vol + tgt_vol - inter_vol
    return float(inter_vol / max(union_vol, 1e-6))

# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train(dataset_root="dataset_aug", batch_size=16, epochs=80, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = OBBDataset(dataset_root, split="train")
    val_dataset = OBBDataset(dataset_root, split="val")
    
    # Reduced batch size to 32 for smaller datasets
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    model = ClusterToOBBModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler based on validation IoU
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion_reg = nn.SmoothL1Loss()
    criterion_yaw = nn.L1Loss()
    
    best_val_iou = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # Statistics arrays
    train_losses = []
    val_ious = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            points = batch["points"].to(device)
            target_center = batch["center"].to(device)
            target_dim = batch["dimensions"].to(device)
            target_yaw = batch["yaw_cos_sin"].to(device)
            
            optimizer.zero_grad()
            preds = model(points)
            
            loss_c = criterion_reg(preds["center"], target_center)
            loss_d = criterion_reg(preds["dimensions"], target_dim)
            loss_y = criterion_yaw(preds["yaw_cos_sin"], target_yaw)
            
            loss = loss_c + loss_d + 0.5 * loss_y
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_iou_sum = 0.0
        samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                points = batch["points"].to(device)
                centroids = batch["centroid"].numpy()
                
                preds = model(points)
                
                pred_c = preds["center"].cpu().numpy() + centroids
                pred_d = preds["dimensions"].cpu().numpy()
                pred_yaw_raw = preds["yaw_cos_sin"].cpu().numpy()
                
                tgt_c = batch["center"].numpy() + centroids
                tgt_d = batch["dimensions"].numpy()
                tgt_yaw_raw = batch["yaw_cos_sin"].numpy()
                
                for i in range(len(points)):
                    pyaw = math.atan2(pred_yaw_raw[i, 1], pred_yaw_raw[i, 0])
                    tyaw = math.atan2(tgt_yaw_raw[i, 1], tgt_yaw_raw[i, 0])
                    
                    iou = compute_3d_iou(pred_c[i], pred_d[i], pyaw, tgt_c[i], tgt_d[i], tyaw)
                    val_iou_sum += iou
                    samples += 1
                    
        avg_train_loss = train_loss / len(train_loader)
        avg_val_iou = val_iou_sum / max(samples, 1)
        
        train_losses.append(avg_train_loss)
        val_ious.append(avg_val_iou)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val 3D IoU: {avg_val_iou:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_val_iou)
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_obb_model.pth")
            print("  -> Saved new best model.")

    # Print Post-Training Statistics
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total Time        : {total_time/60:.2f} minutes")
    print(f"Total Epochs      : {epochs}")
    print(f"Best Validation IoU : {best_val_iou:.4f} (Achieved at Epoch {best_epoch})")
    print(f"Final Train Loss  : {train_losses[-1]:.4f}")
    print(f"Average Train Loss: {sum(train_losses)/len(train_losses):.4f}")
    print("="*50)

if __name__ == "__main__":
    train(dataset_root="dataset_aug", batch_size=32, epochs=100, lr=1e-3)