import librosa
import numpy as np
import os
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

def preprocess_audio(input_dir, output_dir):
    """
    Tiền xử lý các file âm thanh .wav trong thư mục đầu vào và lưu vào thư mục đầu ra.

    Args:
        input_dir (str): Đường dẫn đến thư mục chứa các file âm thanh .wav đầu vào.
        output_dir (str): Đường dẫn đến thư mục chứa các file âm thanh .wav sau khi xử lý.
    """

    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Duyệt qua từng file trong thư mục đầu vào
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Đọc file âm thanh
            y, sr = librosa.load(input_path, sr=None)

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

            # Lưu file âm thanh đã xử lý
            librosa.output.write_wav(output_path, y_clean, sr)

            print(f"Đã xử lý và lưu file: {output_path}")

# Ví dụ sử dụng
input_directory = "path/to/input/directory"  # Thay đổi đường dẫn đầu vào
output_directory = "path/to/output/directory" # Thay đổi đường dẫn đầu ra
preprocess_audio(input_directory, output_directory)