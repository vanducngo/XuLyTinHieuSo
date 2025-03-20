import sounddevice as sd
import numpy as np
import whisper
import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 1. Hàm tiền xử lý âm thanh
def preprocess_audio(y, sr):
    # Chuẩn hóa tín hiệu âm thanh
    y = librosa.util.normalize(y)

    # Loại bỏ đoạn tĩnh (Silence Removal)
    non_silent_indices = librosa.effects.split(y, top_db=30)
    y_trimmed = np.concatenate([y[start:end] for start, end in non_silent_indices])

    # Tính STFT để xử lý nhiễu
    S_full, phase = librosa.magphase(librosa.stft(y_trimmed, n_fft=2048, hop_length=512))
    noise_power = np.mean(S_full[:, :int(sr * 0.1)], axis=1)

    # Áp dụng mặt nạ giảm nhiễu
    mask = S_full > (noise_power[:, None] * 1.5)
    mask = medfilt(mask.astype(float), kernel_size=(1, 5))

    # Tăng cường tín hiệu (Spectral Enhancement)
    S_clean = S_full * mask
    S_clean = gaussian_filter1d(S_clean, sigma=1, axis=0)

    # Tái tạo tín hiệu âm thanh
    y_clean = librosa.istft(S_clean * phase, hop_length=512)

    return y_clean, sr

# 2. Hàm nhận dạng giọng nói với Whisper
def transcribe_audio_from_mic():
    # Thiết lập thông số thu âm
    fs = 16000  # Tần số lấy mẫu
    duration = 5  # Thời gian thu âm (giây)

    print("Đang thu âm...")

    # Thu âm từ mic
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Chờ đến khi thu âm xong

    print("Thu âm xong, đang xử lý...")

    # Tiền xử lý âm thanh
    y_clean, sr = preprocess_audio(audio_data.flatten(), fs)

    # Lưu tệp âm thanh đã xử lý tạm thời
    temp_file = "processed_audio.wav"
    sf.write(temp_file, y_clean, sr, subtype='PCM_16')

    # Tải mô hình Whisper
    model = whisper.load_model("base")  

    # Nhận dạng giọng nói từ tệp âm thanh đã giảm nhiễu
    result = model.transcribe(temp_file)

    # In ra kết quả nhận dạng
    print("Kết quả nhận dạng:", result['text'])

# Gọi hàm nhận dạng giọng nói từ mic
transcribe_audio_from_mic()
