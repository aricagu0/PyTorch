import os, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import DrivingDataset, STEER_SCALE
from model import PilotNet

MODEL_DIR = r'C:\Users\user\Desktop\aaa\models'
DATA_DIR  = r'C:\Users\user\Desktop\training_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PilotNet().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt'), map_location=device))
model.eval()

val_ds = DrivingDataset(DATA_DIR, use_augment=False, split='val')
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

preds, trues = [], []
with torch.no_grad():
    for imgs, steers in val_loader:
        imgs = imgs.to(device)
        out = model(imgs).cpu().numpy() * STEER_SCALE
        preds.extend(out.tolist())
        trues.extend((steers.numpy() * STEER_SCALE).tolist())

preds = np.array(preds)
trues = np.array(trues)
mae  = np.mean(np.abs(preds - trues))
rmse = np.sqrt(np.mean((preds - trues)**2))
print(f'MAE : {mae:.2f}  (평균 오차 steer 단위)')
print(f'RMSE: {rmse:.2f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

with open(os.path.join(MODEL_DIR, 'history.json')) as f:
    history = json.load(f)
axes[0].plot(history['train'], label='Train Loss')
axes[0].plot(history['val'],   label='Val Loss')
axes[0].set_title('Train / Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(trues, preds, alpha=0.3, s=10)
mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
axes[1].plot([mn, mx], [mn, mx], 'r--', label='완벽한 예측선')
axes[1].set_title(f'예측값 vs 실제값  (MAE={mae:.1f})')
axes[1].set_xlabel('실제 steer')
axes[1].set_ylabel('예측 steer')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
out_path = os.path.join(MODEL_DIR, 'evaluate_result.png')
plt.savefig(out_path, dpi=120)
print(f'그래프 저장: {out_path}')
