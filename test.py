import librosa
import numpy as np
from scipy.signal import medfilt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import sounddevice as sd
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Hàm ghi âm từ micro
def record_audio(filename, duration, samplerate=16000):
    print("Bắt đầu ghi âm...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()  # Chờ quá trình ghi âm hoàn tất
    sf.write(filename, audio, samplerate)
    print(f"Ghi âm hoàn tất, tệp đã được lưu: {filename}")

# 2. Hàm tiền xử lý âm thanh
def preprocess_audio(input_file, output_file):
    # Đọc file âm thanh
    y, sr = librosa.load(input_file, sr=None)

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

    # Lưu tệp âm thanh đã xử lý
    sf.write(output_file, y_clean, sr, subtype='PCM_16')

# 3. Hàm chuyển đổi giọng nói thành văn bản
def speech_to_text(input_file, model_name="facebook/wav2vec2-base-960h"):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Đọc file âm thanh đã xử lý
    y, sr = librosa.load(input_file, sr=16000)
    input_values = processor(y, sampling_rate=sr, return_tensors="pt", padding=True).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 4. Main script
if __name__ == "__main__":
    # Ghi âm giọng nói
    recorded_file = "recorded_audio.wav"
    clean_file = "clean_audio.wav"
    duration = 5  # Thời lượng ghi âm (giây)
    record_audio(recorded_file, duration)

    # Tiền xử lý âm thanh
    preprocess_audio(recorded_file, clean_file)

    # Chuyển đổi giọng nói thành văn bản
    result = speech_to_text(clean_file)
    print("Kết quả chuyển đổi giọng nói thành văn bản:")
    print(result)
