import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DrivingDataset
from model import PilotNet

DATA_DIR   = r'C:\Users\user\Desktop\training_data'
MODEL_DIR  = r'C:\Users\user\Desktop\aaa\models'
EPOCHS     = 60
BATCH_SIZE = 32
LR         = 0.001
PATIENCE   = 12

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'학습 장치: {device}')

train_ds = DrivingDataset(DATA_DIR, use_augment=True,  split='train')
val_ds   = DrivingDataset(DATA_DIR, use_augment=False, split='val')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = PilotNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
no_improve = 0
history = {'train': [], 'val': []}

print(f'\n학습 시작 — train:{len(train_ds)}샘플, val:{len(val_ds)}샘플')
print('='*60)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for imgs, steers in train_loader:
        imgs, steers = imgs.to(device), steers.to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = criterion(pred, steers)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, steers in val_loader:
            imgs, steers = imgs.to(device), steers.to(device)
            pred = model(imgs)
            val_loss += criterion(pred, steers).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    saved = ''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
        saved = ' ← 저장!'
    else:
        no_improve += 1

    lr_now = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch:3d}/{EPOCHS} | train:{train_loss:.5f} | val:{val_loss:.5f} | lr:{lr_now:.6f}{saved}')

    if no_improve >= PATIENCE:
        print(f'조기 종료: {PATIENCE}에폭 동안 개선 없음')
        break

print(f'\n학습 완료! 최고 val loss: {best_val_loss:.5f}')
print(f'모델 저장: {MODEL_DIR}/best_model.pt')

import json
with open(os.path.join(MODEL_DIR, 'history.json'), 'w') as f:
 json.dump(history, f)
print('학습 기록 저장: models/history.json')
