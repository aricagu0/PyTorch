import cv2
import os
import sys
import time
import numpy as np

# 화면(디스플레이)이 없는 환경에서도 실행 가능
# --headless 옵션 또는 DISPLAY가 없으면 자동으로 헤드리스 모드
HEADLESS = ('--headless' in sys.argv) or (os.environ.get('DISPLAY', '') == '')
from camera_stream import CameraStream
from serial_link import SerialLink
from control_input import RemoteControlInput
from data_logger import DataLogger

# AI 모드: model.py와 dataset.py가 있을 때만 활성화
AI_AVAILABLE = False
try:
    from model import PilotNet
    from dataset import preprocess, STEER_SCALE
    import torch
    AI_AVAILABLE = True
except ImportError:
    pass

AI_BASE_SPEED = 50

def load_ai_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pt')
    if not os.path.exists(model_path):
        print(f'[AI] 모델 파일 없음: {model_path}')
        return None
    model = PilotNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('[AI] 모델 로드 완료!')
    return model

def ai_predict(model, frame):
    processed = preprocess(frame)
    black_ratio = np.sum(processed > 128) / processed.size
    if black_ratio < 0.03:
        print('[AI 경고] 트랙 미검충! 정지')
        return 0, 0   # 트랙 없으면 정지
    tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        steer_norm = model(tensor).item()
    return AI_BASE_SPEED, int(steer_norm * STEER_SCALE)

def main():
    cam  = CameraStream(index='/dev/video0')
    link = SerialLink(port='/dev/serial0')
    logger = DataLogger()

    ai_model = load_ai_model() if AI_AVAILABLE else None
    ai_mode  = False

    def on_start():
        frame = cam.read()
        link.reset_yaw()  # ★ 녹화 시작 버튼을 누른 이 순간의 각도를 90도로 영점 초기화
        if frame is not None:
            logger.start(frame.shape, fps=30)

    def on_stop():
        logger.stop()

    ctrl = RemoteControlInput(on_start=on_start, on_stop=on_stop)

    last_speed = None
    last_steer = None
    log_counter = 0   # 콘솔 출력 주기 카운터

    print("[조작] 헤드리스 모드: Enter=AI ON/OFF  |  Ctrl+C=종료" if HEADLESS else "[조작] [a] AI자율주행 ON/OFF  |  [q] 종료")

    # 헤드리스 모드에서는 AI 모드 자동 시작
    if HEADLESS and ai_model is not None:
        ai_mode = True
        print('[헤드리스] AI 자율주행 자동 시작!')

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a') and ai_model is not None:
                ai_mode = not ai_mode
                print(f'[모드 전환] {"AI 자율주행 시작!" if ai_mode else "수동 조종 모드"}')
                if not ai_mode:
                    link.set_command(0, 0)

            if ai_mode and ai_model is not None:
                # AI 자율주행 모드
                speed, steer = ai_predict(ai_model, frame)
                if (speed, steer) != (last_speed, last_steer):
                    link.set_command(speed, steer)
                    last_speed, last_steer = speed, steer
                yaw    = link.get_yaw()
                status = f'AI MODE  steer:{steer:+4d}  yaw:{yaw:+6.1f}'
                color  = (0, 255, 0)
            else:
                # 수동 조종 모드 (기존 방식)
                speed, steer, armed = ctrl.read()
                current_speed = speed if armed else 0
                current_steer = steer if armed else 0
                if (current_speed, current_steer) != (last_speed, last_steer):
                    link.set_command(current_speed, current_steer)
                    last_speed = current_speed
                    last_steer = current_steer
                yaw = link.get_yaw()
                if ctrl.recording:
                    logger.log(frame, speed, steer, yaw)
                status = 'REC' if ctrl.recording else 'STANDBY (시작 버튼)'
                color  = (0, 0, 255) if ctrl.recording else (200, 200, 200)

            # ── 콘솔 로그 출력 (30프레임마다 한 번) ──────────────
            log_counter += 1
            if log_counter >= 30:
                log_counter = 0
                mode_str = 'AI' if ai_mode else 'RC'
                print(f'[{mode_str}] SPEED:{speed if ai_mode else current_speed:+4d}  '
                      f'STEER:{steer if ai_mode else current_steer:+4d}  '
                      f'YAW:{yaw:+6.1f}')

            if HEADLESS:
                time.sleep(0.05)  # 50ms 대기 (20fps)
            else:
                ai_label = ' | [a]AI ON' if (ai_model and not ai_mode) else (' | [a]AI OFF' if ai_mode else '')
                cv2.putText(frame, f'{status}{ai_label}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow('RC Car Control', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a') and ai_model is not None:
                    ai_mode = not ai_mode
                    print(f'[모드 전환] {"AI 자율주행 시작!" if ai_mode else "수동 조종 모드"}')
                    if not ai_mode:
                        link.set_command(0, 0)
    finally:
        if ctrl.recording:
            logger.stop()
        link.stop()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()