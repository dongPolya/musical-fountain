import pyaudio
import numpy as np
import pygame
import sys
import random
import json
import os

# ---------- 加载校准数据 ----------
CALIB_FILE = "calibration.json"
calibration = {}
if os.path.exists(CALIB_FILE):
    with open(CALIB_FILE, "r") as f:
        calibration = json.load(f)
    print(f"加载校准数据，共 {len(calibration)} 个音符")
else:
    print("未找到校准文件，使用默认检测")

freq_to_key = {}
for key, data in calibration.items():
    freq = data["frequency"]
    freq_to_key[freq] = key

# ---------- 音频参数 ----------
CHUNK = 2048  # 减小缓冲区，降低延迟
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 150  # 降低阈值，提高灵敏度

# ---------- 定义14个白键 ----------
WHITE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
OCTAVES = [4, 5]
KEY_LIST = []
for oct in OCTAVES:
    for note in WHITE_NOTES:
        KEY_LIST.append((note, oct))
NUM_KEYS = len(KEY_LIST)

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
        for idx, (n, o) in enumerate(KEY_LIST):
            if n == note_name and o == octave:
                return idx
    return None

# ---------- 自相关基频检测（快速版） ----------
def detect_pitch(audio):
    # 归一化
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    corr = np.correlate(audio, audio, mode='full')
    corr = corr[len(corr)//2:]
    min_lag = int(RATE / 2000)
    max_lag = int(RATE / 80)
    if max_lag > len(corr):
        max_lag = len(corr) - 1
    if max_lag <= min_lag:
        return None, 0
    segment = corr[min_lag:max_lag+1]
    idx = np.argmax(segment) + min_lag
    if idx > 0 and idx < len(corr)-1:
        y0, y1, y2 = corr[idx-1], corr[idx], corr[idx+1]
        delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2 + 1e-10)
        idx = idx + delta
    freq = RATE / idx
    amp = corr[int(idx)] if idx < len(corr) else 0
    if freq < 80 or freq > 2000:
        return None, 0
    return freq, amp

# ---------- Pygame 初始化 ----------
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("蓝色音乐喷泉 (快速响应)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 16)

# ---------- 喷泉柱位置 ----------
MARGIN_LEFT, MARGIN_RIGHT = 80, 80
x_positions = {}
for idx, (note, oct) in enumerate(KEY_LIST):
    x = MARGIN_LEFT + idx * (WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (NUM_KEYS - 1)
    x_positions[(note, oct)] = int(x)

energy = {key: 0.0 for key in KEY_LIST}
MAX_HEIGHT = HEIGHT - 80
DECAY_RATE = 0.99
ENERGY_BOOST = 0.3
HUE_BLUE = 0.6

# ---------- 粒子类 ----------
class Particle:
    def __init__(self, key_idx, base_x, base_y, target_height, brightness):
        self.key_idx = key_idx
        self.base_x = base_x
        self.base_y = base_y
        self.target_height = target_height
        self.color = pygame.Color(0,0,0)
        self.color.hsva = (HUE_BLUE*360, 80, brightness, 100)
        self.x = base_x + random.uniform(-50, 50)
        self.vx = random.uniform(-1.5, 1.5)
        self.y = base_y
        # 提高喷射速度，范围 22~38
        self.rise_speed = random.uniform(22, 38)
        self.lifetime = random.randint(40, 80)  # 寿命稍减，让喷泉更灵敏
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
        if alpha < 0: alpha = 0
        self.color.a = int(alpha)
        if self.color.a <= 0 or self.age >= self.lifetime:
            return False
        return True

    def draw(self, screen):
        if self.color.a > 0:
            pos = (int(self.x), int(self.y))
            pygame.draw.circle(screen, self.color, pos, self.size)
            if self.size >= 2:
                glow = pygame.Color(self.color)
                glow.a = self.color.a // 3
                pygame.draw.circle(screen, glow, pos, self.size + 2)

# ---------- 音频初始化 ----------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

particles = []
base_y = HEIGHT - 50
current_note_display = "等待演奏..."

# ---------- 主循环 ----------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        peak_freq, peak_amp = detect_pitch(audio)

        # 只要有有效频率且音量足够，立刻响应，不经过滤波延迟
        if peak_freq is not None and peak_amp > THRESHOLD:
            # 直接使用当前检测频率
            stable_freq = peak_freq

            matched_key = None
            if freq_to_key:
                cal_freqs = list(freq_to_key.keys())
                closest_freq = min(cal_freqs, key=lambda f: abs(f - stable_freq))
                if abs(stable_freq - closest_freq) < 3.0:
                    key_str = freq_to_key[closest_freq]
                    note_name = key_str[0]
                    octave = int(key_str[1])
                    matched_key = (note_name, octave)
            if matched_key is None:
                info = freq_to_note_info(stable_freq)
                if info:
                    note_name, octave, _ = info
                    if note_name in WHITE_NOTES and octave in OCTAVES:
                        matched_key = (note_name, octave)

            if matched_key is not None:
                note, oct = matched_key
                key_idx = get_key_index(note, oct)
                if key_idx is not None:
                    current_note_display = f"{note}{oct}"
                    key = (note, oct)
                    energy[key] = min(1.0, energy[key] + ENERGY_BOOST)
                    center = (NUM_KEYS - 1) / 2.0
                    dist = abs(key_idx - center) / center if center != 0 else 0
                    # 亮度保持边缘更亮（最低40）
                    brightness = 100 - dist * 60
                    brightness = max(40, min(100, brightness))
                    # 增加粒子数量，提升爆发感
                    num_particles = random.randint(60, 100)
                    target_h = MAX_HEIGHT
                    base_x = x_positions[key]
                    for _ in range(num_particles):
                        p = Particle(key_idx, base_x, base_y, target_h, brightness)
                        p.size = random.randint(1, 2)
                        p.vx = random.uniform(-2, 2)
                        particles.append(p)

        # 能量衰减（保持原有逻辑，但不受音符持续影响）
        for key in KEY_LIST:
            energy[key] *= DECAY_RATE
            if energy[key] < 0.001:
                energy[key] = 0.0

    except Exception as e:
        print("音频错误:", e)

    # 更新粒子
    particles = [p for p in particles if p.update()]

    # 绘制画面
    screen.fill((10, 10, 30))

    # 底部标记
    for idx, (note, oct) in enumerate(KEY_LIST):
        x = x_positions[(note, oct)]
        center = (NUM_KEYS - 1) / 2.0
        dist = abs(idx - center) / center if center != 0 else 0
        brightness = 100 - dist * 60
        brightness = max(40, min(100, brightness))
        color = pygame.Color(0,0,0)
        color.hsva = (HUE_BLUE*360, 80, brightness, 100)
        pygame.draw.line(screen, color, (x, base_y-5), (x, base_y+5), 4)

    # 绘制所有粒子
    for p in particles:
        p.draw(screen)

    # 显示当前音符
    screen.blit(font.render(current_note_display, True, (220,220,220)), (20,20))

    # 能量条（显示前6个键）
    for i, key in enumerate(KEY_LIST[:6]):
        e = energy[key]
        bar_x = 20 + i*90
        pygame.draw.rect(screen, (50,50,50), (bar_x, 60, 70, 10))
        pygame.draw.rect(screen, (100,200,255), (bar_x, 60, 70*e, 10))

    fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, (150,150,150))
    screen.blit(fps_text, (WIDTH-100, 20))

    pygame.display.flip()
    clock.tick(60)

# 清理资源
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()