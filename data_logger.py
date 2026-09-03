import os, csv, time, cv2

class DataLogger:
    def __init__(self, base_dir='training_data'):
        self.base_dir = base_dir
        self.session_dir = None
        self.img_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.frame_idx = 0
        self.start_time = None
        self.frame_shape = None

    def start(self, frame_shape, fps=30):
        session_name = time.strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.base_dir, session_name)
        self.img_dir = os.path.join(self.session_dir, 'IMG')
        os.makedirs(self.img_dir, exist_ok=True)

        self.csv_file = open(os.path.join(self.session_dir, 'driving_log.csv'),
                              'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'image_file', 'speed', 'steer','yaw'])

        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.session_dir, 'session.avi'), fourcc, fps, (w, h))
        self.frame_idx = 0
        self.start_time = time.time()
        print(f"[REC 시작] {self.session_dir}")

    def log(self, frame, speed, steer,yaw):
        if self.csv_writer is None:
            return
        ts = time.time()
        img_name = f'trin_{self.frame_idx:06d}_{int(yaw):03d}.jpg'
        cv2.imwrite(os.path.join(self.img_dir, img_name), frame)
        self.csv_writer.writerow([f'{ts:.3f}', img_name, speed, steer,yaw])
        self.video_writer.write(frame)
        self.frame_idx += 1

    def stop(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
            # ★ 실제 재생 시간에 맞춰 30fps로 프레임 복제 (너무 빠르게 재생되는 현상 완벽 방지)
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.frame_idx > 0:
                real_fps = self.frame_idx / elapsed
                print(f"[REC 종료] 총 {self.frame_idx}프레임, 실제 {elapsed:.1f}초 -> 실측 fps={real_fps:.2f}")
                import threading
                t = threading.Thread(target=self._refix_video_fps, args=(real_fps, elapsed))
                t.start()
        print(f"[REC 종료] -> {self.session_dir} (백그라운드에서 인코딩 중...)")

    def _refix_video_fps(self, real_fps, elapsed):
        old_path = os.path.join(self.session_dir, 'session.avi')
        tmp_path = os.path.join(self.session_dir, 'session_fixed.avi')
        cap = cv2.VideoCapture(old_path)
        h, w = self.frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # 무조건 호환성이 가장 좋은 30.0 fps로 고정하여 파일 생성
        writer = cv2.VideoWriter(tmp_path, fourcc, 30.0, (w, h))
        
        target_frames = int(elapsed * 30.0)
        cap_idx = 0
        ok, frame = cap.read()
        
        for out_idx in range(target_frames):
            expected_time = out_idx / 30.0
            
            # 다음 프레임으로 넘어갈 시간이 되었다면 프레임 전진
            while ((cap_idx + 1) / real_fps) <= expected_time:
                next_ok, next_frame = cap.read()
                if not next_ok:
                    break
                frame = next_frame
                cap_idx += 1
                ok = next_ok
                
            if frame is not None:
                writer.write(frame)
                
        cap.release()
        writer.release()
        os.replace(tmp_path, old_path)   # 원본을 30fps 고정 버전으로 교체