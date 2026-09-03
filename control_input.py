import threading, select, logging, time
from evdev import InputDevice, ecodes, list_devices

# 디버깅을 위한 파일 로거 설정
logging.basicConfig(
    filename='remote_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

SPEED_STEP = 5
STEER_STEP = 10
START_SPEED = 50

class RemoteControlInput:
    def __init__(self, device_path=None, on_start=None, on_stop=None):
        self.dev = InputDevice(device_path or self._find_ir_device())
        self.speed = 0
        self.steer = 0
        self.armed = False
        self.recording = False
        self.on_start = on_start
        self.on_stop = on_stop
        self.lock = threading.Lock()
        
        logging.info("RemoteControlInput 초기화 완료. 이벤트 수신 대기 시작.")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _find_ir_device(self):
        for path in list_devices():
            d = InputDevice(path)
            if 'ir' in d.name.lower() or 'rc' in d.name.lower():
                return path
        raise RuntimeError("IR 입력장치를 못 찾았습니다.")

    def _loop(self):                          # ★ 여기, 클래스 안쪽 들여쓰기 확인
        while self.running:
            r, _, _ = select.select([self.dev.fd], [], [], 0.05)
            if not r:
                continue

            events = list(self.dev.read())
            
            # 읽어온 모든 키 이벤트를 원본 그대로 로깅
            for e in events:
                if e.type == ecodes.EV_KEY:
                    logging.debug(f"[EVDEV RAW] type={e.type}, code={e.code}, value={e.value} (1:눌림, 0:뗌, 2:반복)")

            stop_events = [e for e in events if e.type == ecodes.EV_KEY
                           and e.code == ecodes.KEY_STOP and e.value == 1]
            
            if stop_events:
                logging.warning(f"!!! STOP 버튼 눌림 감지 !!! (armed 해제 및 정지 명령 대기)")
                need_stop = False
                with self.lock:
                    self.armed = False
                    self.speed = 0
                    self.steer = 0
                    
                    if self.recording:
                        self.recording = False
                        need_stop = True
                
                # Lock 블록 밖에서 on_stop을 호출해야 메인 스레드가 멈추는 데드락을 방지할 수 있습니다.
                if need_stop and self.on_stop:
                    self.on_stop()
                continue
            # ★ 같은 코드의 이벤트는 이번 배치에서 몇 번 나왔는지 개수만 세고,
            #    그 횟수만큼만 STEP을 한 번에 계산해서 적용 (반복 적용 오버헤드 제거)
            key_counts = {}
            ok_pressed = False
            for e in events:
                if e.type != ecodes.EV_KEY or e.value not in (1, 2):
                    continue
                if e.code == ecodes.KEY_OK and e.value == 1:
                    ok_pressed = True
                    continue
                key_counts[e.code] = key_counts.get(e.code, 0) + 1

            with self.lock:
                if ok_pressed and not self.recording:
                    self.armed = True
                    self.recording = True
                    self.speed = START_SPEED
                    self.steer = 0
                    if self.on_start:
                        self.on_start()

                if self.armed:
                    if ecodes.KEY_REWIND in key_counts:
                        self.speed = max(-100, self.speed - SPEED_STEP * key_counts[ecodes.KEY_REWIND])
                    if ecodes.KEY_FASTFORWARD in key_counts:
                        self.speed = min(100, self.speed + SPEED_STEP * key_counts[ecodes.KEY_FASTFORWARD])
                    if ecodes.KEY_KPMINUS in key_counts:
                        self.steer = max(-100, self.steer - STEER_STEP * key_counts[ecodes.KEY_KPMINUS])
                    if ecodes.KEY_KPPLUS in key_counts:
                        self.steer = min(100, self.steer + STEER_STEP * key_counts[ecodes.KEY_KPPLUS])
                        logging.info(f"방향 전환 (우): 현재 STEER={self.steer}")

    def read(self):
        with self.lock:
            return self.speed, self.steer, self.armed

    def stop(self):
        self.running = False