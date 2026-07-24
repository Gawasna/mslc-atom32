import numpy as np

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_banks(num_bins, fft_len, sample_rate, low_freq=20.0, high_freq=None):
    if high_freq is None:
        high_freq = sample_rate / 2.0
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((fft_len + 1) * hz_points / sample_rate).astype(int)
    banks = np.zeros((num_bins, fft_len // 2 + 1))
    for m in range(1, num_bins + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m - f_m_minus > 0:
                banks[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus - f_m > 0:
                banks[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return banks

def compute_fbank(audio, sample_rate=16000, num_mel_bins=80, frame_length_ms=25.0, frame_shift_ms=10.0):
    alpha = 0.97
    audio = np.append(audio[0], audio[1:] - alpha * audio[:-1])
    frame_length = int(round(sample_rate * frame_length_ms / 1000.0))
    frame_shift = int(round(sample_rate * frame_shift_ms / 1000.0))
    num_samples = len(audio)
    if num_samples < frame_length:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    num_frames = 1 + int(np.floor((num_samples - frame_length) / frame_shift))
    window = np.hamming(frame_length).astype(np.float32)
    fft_len = 512
    mel_banks = get_mel_banks(num_mel_bins, fft_len, sample_rate)
    features = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        frame = audio[start:end] * window
        fft_complex = np.fft.rfft(frame, n=fft_len)
        power_spectrum = np.abs(fft_complex)**2 / fft_len
        mel_energies = np.dot(mel_banks, power_spectrum)
        log_mel_energies = np.log(np.maximum(mel_energies, 1e-10))
        features.append(log_mel_energies)
    features = np.array(features, dtype=np.float32)
    if len(features) > 0:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-5
        features = (features - mean) / std
    return features
