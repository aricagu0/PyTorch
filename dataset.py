import os, csv, random, numpy as np, cv2, torch
from torch.utils.data import Dataset

CROP_TOP = 0.45
IMG_W, IMG_H = 200, 66
STEER_SCALE = 100.0
STRAIGHT_KEEP_RATIO = 0.35

def preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    crop = img_bgr[int(h * CROP_TOP):, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    return cv2.resize(binary, (IMG_W, IMG_H))

def augment(img, steer):
    r = [(img, steer), (cv2.flip(img, 1), -steer)]
    for d in [-25, 25]:
        b = np.clip(img.astype(np.int16) + d, 0, 255).astype(np.uint8)
        r += [(b, steer), (cv2.flip(b, 1), -steer)]
    return r

class DrivingDataset(Dataset):
    def __init__(self, sessions_dir, use_augment=True, split='train', val_ratio=0.2, seed=42):
        self.use_augment = use_augment and (split == 'train')
        raw = []
        for sess in sorted(os.listdir(sessions_dir)):
            sp = os.path.join(sessions_dir, sess)
            if not os.path.isdir(sp): continue
            cp = os.path.join(sp, 'driving_log.csv')
            ip = os.path.join(sp, 'IMG')
            if not os.path.exists(cp): continue
            with open(cp, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    fp = os.path.join(ip, row['image_file'])
                    if not os.path.exists(fp): continue
                    try: raw.append((fp, float(row['steer'])))
                    except: pass
        straight = [(p,s) for p,s in raw if s == 0.0]
        others   = [(p,s) for p,s in raw if s != 0.0]
        random.seed(seed)
        balanced = random.sample(straight, int(len(straight)*STRAIGHT_KEEP_RATIO)) + others
        random.shuffle(balanced)
        n_val = int(len(balanced) * val_ratio)
        self.samples = balanced[:n_val] if split == 'val' else balanced[n_val:]
        sc = {}
        for _,s in self.samples: sc[int(s)] = sc.get(int(s),0)+1
        print(f'[Dataset] {split}: {len(self.samples)}샘플 | {dict(sorted(sc.items()))}')

    def __len__(self):
        return len(self.samples)*5 if self.use_augment else len(self.samples)

    def __getitem__(self, idx):
        if self.use_augment:
            img_path, steer = self.samples[idx//5]
            img, steer = augment(preprocess(cv2.imread(img_path)), steer)[idx%5]
        else:
            img_path, steer = self.samples[idx]
            img = preprocess(cv2.imread(img_path))
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)/255.0, torch.tensor(steer/STEER_SCALE, dtype=torch.float32)
