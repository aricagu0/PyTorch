import serial, struct, threading, time, csv, os
from datetime import datetime

STX = 0xAA
CMD_DRIVE = 0x01
MSG_TELEMETRY = 0x02

class SerialLink:
    def __init__(self, port='/dev/serial0', baud=115200, hz=30, log_dir='comm_logs'):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.speed = 0
        self.steer = 0
        self.yaw_raw = 0.0
        self.yaw_offset = 0.0
        self.lock = threading.Lock()
        self.running = True
        self.period = 1.0 / hz

        # ── 통신 로그 파일 초기화 ─────────────────────────
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f'comm_{ts}.csv')
        self._log_file = open(log_path, 'w', newline='', encoding='utf-8')
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(['timestamp', 'direction', 'speed', 'steer', 'yaw'])
        self._log_file.flush()
        print(f'[SerialLink] 통신 로그 저장: {log_path}')
        # ─────────────────────────────────────────────────

        self.tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.tx_thread.start()
        self.rx_thread.start()

    def set_command(self, speed, steer):
        with self.lock:
            self.speed = max(-100, min(100, speed))
            self.steer = max(-100, min(100, steer))

    def get_yaw(self):
        with self.lock:
            # 최종 결과값을 360으로 나눈 나머지로 반환 (항상 0.0 ~ 359.9... 유지)
            return (self.yaw_raw - self.yaw_offset + 90.0) % 360.0

    def reset_yaw(self):
        with self.lock:
            self.yaw_offset = self.yaw_raw

    def _tx_loop(self):
        while self.running:
            with self.lock:
                s, st = self.speed, self.steer
            frame = self._build_drive_frame(s, st)
            self.ser.write(frame)

            # TX 로그 기록 (speed/steer 명령 전송)
            self._write_log('TX', s, st, '')

            time.sleep(self.period)

    def _build_drive_frame(self, speed, steer):
        payload = struct.pack('<bb', speed, steer)
        chk = CMD_DRIVE ^ len(payload)
        for b in payload:
            chk ^= b
        return bytes([STX, CMD_DRIVE, len(payload)]) + payload + bytes([chk])

    def _rx_loop(self):
        buf = bytearray()
        while self.running:
            data = self.ser.read(64)
            if data:
                buf += data
                self._parse_buffer(buf)

    def _parse_buffer(self, buf):
        while len(buf) >= 8:
            if buf[0] != STX:
                buf.pop(0)
                continue
            msg_id, length = buf[1], buf[2]
            if msg_id == MSG_TELEMETRY and length == 4 and len(buf) >= 8:
                payload = bytes(buf[3:7])
                chk = buf[1] ^ buf[2]
                for b in payload:
                    chk ^= b
                if chk == buf[7]:
                    yaw = struct.unpack('<f', payload)[0]
                    with self.lock:
                        self.yaw_raw = yaw

                    # RX 로그 기록 (yaw 수신)
                    self._write_log('RX', '', '', round(yaw, 2))

                del buf[:8]
            else:
                buf.pop(0)

    def _write_log(self, direction, speed, steer, yaw):
        """CSV에 한 행 기록 (스레드 안전)"""
        try:
            ts = time.time()
            self._log_writer.writerow([f'{ts:.3f}', direction, speed, steer, yaw])
            self._log_file.flush()   # 즉시 디스크에 기록 (프로그램 종료돼도 손실 없음)
        except Exception:
            pass

    def stop(self):
        self.set_command(0, 0)
        time.sleep(self.period * 2)
        self.running = False
        self.tx_thread.join()
        self.rx_thread.join()
        self.ser.close()
        # 로그 파일 닫기
        try:
            self._log_file.close()
            print('[SerialLink] 통신 로그 저장 완료')
        except Exception:
            pass