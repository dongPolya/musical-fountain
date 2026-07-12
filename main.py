import pyaudio
import numpy as np
import pygame
import sys
import random

# ---------- 音频参数 ----------
CHUNK = 2048                     # 低延迟
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 150                  # 提高灵敏度

# ---------- 音符映射 ----------
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_HUES = {
    'C': 0.00, 'C#': 0.08, 'D': 0.16, 'D#': 0.25, 'E': 0.33, 'F': 0.42,
    'F#': 0.50, 'G': 0.58, 'G#': 0.66, 'A': 0.75, 'A#': 0.83, 'B': 0.92
}

def freq_to_note_info(freq):
    """根据频率返回 (音符名, 八度, 频率)"""
    if freq < 20:
        return None
    midi = 69 + 12 * np.log2(freq / 440.0)
    midi_round = int(round(midi))
    if midi_round < 0 or midi_round > 127:
        return None
    octave = midi_round // 12 - 1
    note_idx = midi_round % 12
    return NOTE_NAMES[note_idx], octave, freq

# ---------- 自相关基频检测（替代 FFT） ----------
def detect_pitch(audio):
    """
    使用自相关（Auto-correlation）检测基频，对音乐信号更鲁棒
    """
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

# ---------- Pygame 初始化 ----------
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("彩色音乐喷泉 (自相关版)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)
small_font = pygame.font.SysFont("Arial", 16)

# ---------- 音阶位置与状态 ----------
MARGIN_LEFT = 80
MARGIN_RIGHT = 80
x_positions = {}
num_notes = len(NOTE_NAMES)
for i, name in enumerate(NOTE_NAMES):
    x = MARGIN_LEFT + i * (WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (num_notes - 1)
    x_positions[name] = int(x)

energy = {name: 0.0 for name in NOTE_NAMES}
MAX_HEIGHT = HEIGHT - 80
DECAY_RATE = 0.99
ENERGY_BOOST = 0.3

# ---------- 粒子类 ----------
class Particle:
    def __init__(self, note_name, base_x, base_y, target_height, color):
        self.note_name = note_name
        self.base_x = base_x
        self.base_y = base_y
        self.target_height = target_height
        self.color = color
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

    # --- 音频处理（自相关） ---
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        peak_freq, peak_amp = detect_pitch(audio)

        # 能量衰减（先衰减）
        for name in NOTE_NAMES:
            energy[name] *= DECAY_RATE
            if energy[name] < 0.001:
                energy[name] = 0.0

        # 如果检测到有效频率且音量足够
        if peak_freq is not None and peak_amp > THRESHOLD:
            info = freq_to_note_info(peak_freq)
            if info:
                name, octave, _ = info
                current_note_display = f"{name}{octave}"
                energy[name] = min(1.0, energy[name] + ENERGY_BOOST)

                # 生成粒子
                num_particles = random.randint(50, 80)
                target_h = MAX_HEIGHT
                hue = NOTE_HUES.get(name, 0.5)
                base_color = pygame.Color(0, 0, 0)
                base_color.hsva = (hue * 360, 80, 100, 100)
                for _ in range(num_particles):
                    color = pygame.Color(base_color)
                    lightness = random.randint(70, 100)
                    color.hsva = (hue * 360, 80, lightness, 100)
                    p = Particle(name, x_positions[name], base_y, target_h, color)
                    p.rise_speed = random.uniform(10, 22)
                    p.size = random.randint(1, 2)
                    p.vx = random.uniform(-2, 2)
                    particles.append(p)

    except Exception as e:
        print("音频错误:", e)

    # --- 更新粒子 ---
    particles = [p for p in particles if p.update()]

    # --- 绘制 ---
    screen.fill((10, 10, 30))

    # 底部彩色标记
    for i, name in enumerate(NOTE_NAMES):
        x = x_positions[name]
        color = pygame.Color(0, 0, 0)
        color.hsva = (NOTE_HUES[name] * 360, 80, 60, 100)
        pygame.draw.line(screen, color, (x, base_y - 5), (x, base_y + 5), 4)

    # 绘制粒子
    for p in particles:
        p.draw(screen)

    # 显示当前音符
    screen.blit(font.render(current_note_display, True, (220, 220, 220)), (20, 20))

    # 能量条（显示前6个音符）
    for i, name in enumerate(NOTE_NAMES[:6]):
        e = energy[name]
        bar_x = 20 + i * 90
        bar_y = 60
        bar_width = 70
        bar_height = 10
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # 能量条颜色跟随音符色调
        bar_color = pygame.Color(0, 0, 0)
        bar_color.hsva = (NOTE_HUES[name] * 360, 80, 70, 100)
        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, bar_width * e, bar_height))

    fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, (150, 150, 150))
    screen.blit(fps_text, (WIDTH - 100, 20))

    pygame.display.flip()
    clock.tick(60)

# 清理
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()