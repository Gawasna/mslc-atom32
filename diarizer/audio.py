import numpy as np
import soundfile as sf
import scipy.signal

def load_audio(file_path, target_sr=16000):
    data, sr = sf.read(file_path, dtype='float32')
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
    return data.astype(np.float32)
