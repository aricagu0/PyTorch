# ===================================================
# auto_drive.py — 라즈베리파이 자율주행 추론 모듈
# train_model.py로 학습 완료 후 이 파일을 라즈베리파이에 복사해서 사용
# ===================================================
import cv2, torch, numpy as np, time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model import PilotNet
from dataset import preprocess, STEER_SCALE

MODEL_PATH  = 'models/best_model.pt'
BASE_SPEED  = 50
device = torch.device('cpu')  # 라즈베리파이는 CPU만 사용

def load_model(path=MODEL_PATH):
    model = PilotNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f'[AI] 모델 로드 완료: {path}')
    return model

def predict_steer(model, frame_bgr):
    processed = preprocess(frame_bgr)
    tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        steer_norm = model(tensor).item()
    steer = int(steer_norm * STEER_SCALE)
    steer = max(-100, min(100, steer))
    return steer

class AutoPilot:
    def __init__(self, serial_link, camera):
        self.model = load_model()
        self.link  = serial_link
        self.cap   = camera
        self.running = False

    def start(self):
        self.running = True
        print('[AutoPilot] 자율주행 시작!')
        t_prev = time.time()
        frame_count = 0

        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                print('[AutoPilot] 카메라 오류!')
                break

            steer = predict_steer(self.model, frame)
            self.link.set_command(BASE_SPEED, steer)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - t_prev)
                print(f'[AutoPilot] steer={steer:+4d} | fps={fps:.1f}')
                t_prev = time.time()

    def stop(self):
        self.running = False
        self.link.set_command(0, 0)
        print('[AutoPilot] 자율주행 정지!')


if __name__ == '__main__':
    # 단독 실행 시 테스트 (라즈베리파이에서 main.py와 연동)
    model = load_model()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('카메라 스트림 테스트 (q키 종료)')
    while True:
        ok, frame = cap.read()
        if not ok: break
        steer = predict_steer(model, frame)
        cv2.putText(frame, f'steer={steer:+d}', (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.imshow('AutoPilot Preview', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
