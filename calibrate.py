import pyaudio
import numpy as np
import pygame
import json
import os
import sys
import time
import threading
import queue
import random
from collections import deque

# ---------- 公共参数 ----------
WHITE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
OCTAVES = [4, 5]
KEY_LIST = []
for oct in OCTAVES:
    for note in WHITE_NOTES:
        KEY_LIST.append(f"{note}{oct}")

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 200  # 降低阈值，提高灵敏度

# ---------- 音符映射 ----------
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note_info(freq):
    if freq < 20:
        return None
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_round = int(round(midi))
    if midi_round < 0 or midi_round > 127:
        return None
    octave = midi_round // 12 - 1
    note_idx = midi_round % 12
    return NOTE_NAMES[note_idx], octave, freq

def get_key_index(note_name, octave):
    if note_name in WHITE_NOTES and octave in OCTAVES:
        for idx, key in enumerate(KEY_LIST):
            if key[0] == note_name and int(key[1:]) == octave:
                return idx
    return None

# ---------- 自相关基频检测 ----------
def detect_pitch(audio):
    # 归一化，避免数值问题
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    # 计算自相关
    corr = np.correlate(audio, audio, mode='full')
    corr = corr[len(corr)//2:]  # 取正半轴
    # 搜索延迟范围：对应频率 80~2000 Hz
    min_lag = int(RATE / 2000)
    max_lag = int(RATE / 80)
    if max_lag > len(corr):
        max_lag = len(corr) - 1
    if max_lag <= min_lag:
        return None, 0
    # 在范围内寻找最大自相关值（排除零延迟）
    segment = corr[min_lag:max_lag+1]
    idx = np.argmax(segment) + min_lag
    # 抛物线插值提高精度
    if idx > 0 and idx < len(corr)-1:
        y0, y1, y2 = corr[idx-1], corr[idx], corr[idx+1]
        delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-10)
        idx = idx + delta
    freq = RATE / idx
    amp = corr[int(idx)] if idx < len(corr) else 0
    if freq < 80 or freq > 2000:
        return None, 0
    return freq, amp

def load_calibration():
    if os.path.exists("calibration.json"):
        with open("calibration.json", "r") as f:
            return json.load(f)
    return {}

def save_calibration(calibration):
    with open("calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

# ---------- Pygame 可视化 ----------
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("校准可视化 - 蓝色喷泉 (自相关)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 18)

MARGIN_LEFT, MARGIN_RIGHT = 80, 80
NUM_KEYS = len(KEY_LIST)
x_positions = {}
for idx, key in enumerate(KEY_LIST):
    x = MARGIN_LEFT + idx * (WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (NUM_KEYS - 1)
    x_positions[key] = int(x)

HUE_BLUE = 0.6

# ---------- 粒子类 ----------
class Particle:
    def __init__(self, base_x, base_y, target_height, brightness):
        self.base_x = base_x
        self.base_y = base_y
        self.target_height = target_height
        self.color = pygame.Color(0, 0, 0)
        self.color.hsva = (HUE_BLUE * 360, 80, brightness, 100)
        self.x = base_x + random.uniform(-50, 50)
        self.vx = random.uniform(-1.5, 1.5)
        self.y = base_y
        self.rise_speed = random.uniform(8, 18)
        self.lifetime = random.randint(60, 120)
        self.age = 0
        self.size = random.randint(1, 2)

    def update(self):
        self.age += 1
        self.x += self.vx
        self.vx += random.uniform(-0.2, 0.2)
        if self.x < 20 or self.x > WIDTH - 20:
            self.vx *= -0.5
        self.y -= self.rise_speed
        if self.y <= self.base_y - self.target_height:
            self.y = self.base_y - self.target_height
            self.age += 2
        alpha = 255 * (1 - self.age / self.lifetime)
        if alpha < 0:
            alpha = 0
        self.color.a = int(alpha)
        return self.color.a > 0 and self.age < self.lifetime

    def draw(self, screen):
        if self.color.a > 0:
            pos = (int(self.x), int(self.y))
            pygame.draw.circle(screen, self.color, pos, self.size)
            if self.size >= 2:
                glow = pygame.Color(self.color)
                glow.a = self.color.a // 3
                pygame.draw.circle(screen, glow, pos, self.size + 2)

# ---------- 音频流 ----------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

# ---------- 全局状态 ----------
particles = []
base_y = HEIGHT - 50
MAX_HEIGHT = HEIGHT - 100
current_key = None
recording = False
record_start_time = 0
auto_mode = False
auto_index = 0
freq_display = "等待演奏..."
calibration_data = load_calibration()
last_freq = None
freq_buffer = deque(maxlen=5)

cmd_queue = queue.Queue()

def input_thread():
    while True:
        try:
            cmd = input()
            cmd_queue.put(cmd)
        except:
            break

# ---------- 自动模式 ----------
def start_auto_record():
    global current_key, recording, record_start_time, auto_index, freq_display
    if auto_index >= len(KEY_LIST):
        print("所有音符校准完成！")
        return
    current_key = KEY_LIST[auto_index]
    recording = True
    record_start_time = time.time()
    freq_display = f"录制中: {current_key}"
    print(f"请弹奏 '{current_key}'，按 空格键 停止录制并保存...")
    particles.clear()
    auto_index += 1

def stop_record_and_save():
    global recording, current_key, freq_display, last_freq
    if not recording or current_key is None:
        return
    if last_freq is not None:
        calibration_data[current_key] = {"frequency": float(last_freq), "amplitude": 500}
        save_calibration(calibration_data)
        print(f"  已保存: {current_key} -> {last_freq:.2f} Hz")
    else:
        print(f"  未检测到有效频率，跳过")
    recording = False
    current_key = None
    freq_display = "等待演奏..."
    particles.clear()
    freq_buffer.clear()
    if auto_mode:
        if auto_index < len(KEY_LIST):
            start_auto_record()
        else:
            print("所有音符校准完成！")

# ---------- 主循环 ----------
def main_loop():
    global recording, current_key, freq_display, particles, auto_index, last_freq, auto_mode
    running = True

    input_thread_obj = threading.Thread(target=input_thread, daemon=True)
    input_thread_obj.start()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if recording:
                        stop_record_and_save()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 处理命令队列
        try:
            cmd = cmd_queue.get_nowait()
            if cmd.strip().lower() == 'q':
                running = False
                break
            if not auto_mode:
                key = cmd.strip()
                if key in KEY_LIST:
                    current_key = key
                    recording = True
                    record_start_time = time.time()
                    freq_display = f"录制中: {key}"
                    print(f"开始录制 '{key}'，按空格键停止...")
                    particles.clear()
                    freq_buffer.clear()
                else:
                    print(f"无效音符: {key}，请从 {KEY_LIST} 中选择")
        except queue.Empty:
            pass

        # 音频处理
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            peak_freq, peak_amp = detect_pitch(audio)
            if peak_freq is not None and peak_amp > THRESHOLD:
                freq_buffer.append(peak_freq)
                if len(freq_buffer) >= 3:
                    stable_freq = np.median(list(freq_buffer))
                else:
                    stable_freq = peak_freq
                last_freq = stable_freq
                info = freq_to_note_info(stable_freq)
                if info:
                    note, oct, _ = info
                    freq_display = f"{note}{oct}  {stable_freq:.1f}Hz"
                    # 调试输出，方便观察
                    print(f"检测: {stable_freq:.1f} Hz -> {note}{oct}", end='\r')

                # 录制期间生成喷泉
                if recording and current_key is not None and peak_amp > THRESHOLD:
                    target_h = MAX_HEIGHT
                    base_x = x_positions[current_key]
                    key_idx = get_key_index(current_key[0], int(current_key[1:]))
                    if key_idx is not None:
                        center = (NUM_KEYS - 1) / 2.0
                        dist = abs(key_idx - center) / center if center != 0 else 0
                        brightness = 100 - dist * 50
                        brightness = max(50, min(100, brightness))
                        num_particles = random.randint(30, 60)
                        for _ in range(num_particles):
                            p = Particle(base_x, base_y, target_h, brightness)
                            p.rise_speed = random.uniform(10, 22)
                            p.size = random.randint(1, 2)
                            p.vx = random.uniform(-2, 2)
                            particles.append(p)
        except Exception as e:
            print("音频错误:", e)

        # 更新粒子
        particles = [p for p in particles if p.update()]

        # 绘制
        screen.fill((10, 10, 30))

        for idx, key in enumerate(KEY_LIST):
            x = x_positions[key]
            center = (NUM_KEYS - 1) / 2.0
            dist = abs(idx - center) / center if center != 0 else 0
            brightness = 100 - dist * 60
            brightness = max(40, min(100, brightness))
            color = pygame.Color(0,0,0)
            color.hsva = (HUE_BLUE*360, 80, brightness, 100)
            pygame.draw.line(screen, color, (x, base_y-5), (x, base_y+5), 4)

        if current_key and current_key in x_positions:
            x = x_positions[current_key]
            color = pygame.Color(255, 255, 255)
            pygame.draw.line(screen, color, (x, base_y-10), (x, base_y+10), 2)

        for p in particles:
            p.draw(screen)

        mode_text = "自动模式" if auto_mode else "手动模式"
        info_line = f"{mode_text}  当前校准: {current_key if current_key else '无'}"
        screen.blit(font.render(info_line, True, (200,200,200)), (20,20))
        screen.blit(font.render(freq_display, True, (255,255,200)), (20,60))
        calibrated = len(calibration_data)
        screen.blit(small_font.render(f"已校准: {calibrated}/{len(KEY_LIST)}", True, (150,200,150)), (20, 100))

        if not auto_mode and not recording:
            tip = "在命令行输入音符 (如 C4) 开始录制，按空格键停止"
        elif not auto_mode and recording:
            tip = "按 空格键 停止录制并保存"
        else:
            tip = "自动校准中... 按空格键停止当前录制并保存，自动进入下一个"
        screen.blit(small_font.render(tip, True, (150,150,150)), (20, HEIGHT-30))

        screen.blit(small_font.render(f"FPS: {int(clock.get_fps())}", True, (150,150,150)), (WIDTH-120,20))

        pygame.display.flip()
        clock.tick(60)

    # 清理
    try:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
    except:
        pass
    try:
        p.terminate()
    except:
        pass
    pygame.quit()
    sys.exit()

# ---------- 启动 ----------
def start_calibrator(mode):
    global auto_mode, auto_index
    auto_mode = (mode == 'auto')
    if auto_mode:
        start_auto_record()
    else:
        print("手动模式：在命令行输入音符，按回车开始录制，按空格键停止并保存")
        print("输入 'q' 退出")
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n用户中断")
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except:
            pass
        try:
            p.terminate()
        except:
            pass
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    print("=== 钢琴音符校准工具 (自相关检测) ===")
    print("选择模式:")
    print("1. 自动校准 (用户按顺序弹奏，按空格键确认每个音)")
    print("2. 手动校准 (输入音符，按空格键确认)")
    choice = input("请输入 1 或 2: ").strip()
    if choice == '1':
        start_calibrator('auto')
    elif choice == '2':
        start_calibrator('manual')
    else:
        print("无效选择")
        sys.exit()