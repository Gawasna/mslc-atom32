import os
import time
import numpy as np
import soundfile as sf
from diarizer import LocalSpeakerDiarizer, load_audio

def generate_mixed_audio(spk1_path, spk2_path, out_path, segment_len_sec=4.0, silence_len_sec=1.0, sr=16000):
    audio1 = load_audio(spk1_path, target_sr=sr)
    audio2 = load_audio(spk2_path, target_sr=sr)
    
    seg_samples = int(segment_len_sec * sr)
    silence_samples = int(silence_len_sec * sr)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    mixed_audio = []
    ground_truth = []
    current_time = 0.0
    
    s1 = audio1[0:seg_samples]
    mixed_audio.append(s1)
    ground_truth.append((current_time, current_time + segment_len_sec, "Female (Spk1)"))
    current_time += segment_len_sec
    
    mixed_audio.append(silence)
    current_time += silence_len_sec
    
    s2 = audio2[0:seg_samples]
    mixed_audio.append(s2)
    ground_truth.append((current_time, current_time + segment_len_sec, "Male (Spk2)"))
    current_time += segment_len_sec
    
    mixed_audio.append(silence)
    current_time += silence_len_sec
    
    s3 = audio1[seg_samples:seg_samples * 2]
    mixed_audio.append(s3)
    ground_truth.append((current_time, current_time + segment_len_sec, "Female (Spk1)"))
    current_time += segment_len_sec
    
    mixed_audio.append(silence)
    current_time += silence_len_sec
    
    s4 = audio2[seg_samples:seg_samples * 2]
    mixed_audio.append(s4)
    ground_truth.append((current_time, current_time + segment_len_sec, "Male (Spk2)"))
    current_time += segment_len_sec
    
    full_audio = np.concatenate(mixed_audio)
    sf.write(out_path, full_audio, sr)
    print(f"Generated test mixed audio file: {out_path}")
    print("Ground Truth Speaker Turns:")
    for start, end, label in ground_truth:
        print(f"  [{start:.1f}s - {end:.1f}s] {label}")
        
    return full_audio, ground_truth

def run_diarization_demo():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    spk1_path = os.path.join(models_dir, "spk1.wav")
    spk2_path = os.path.join(models_dir, "spk2.wav")
    mixed_path = os.path.join(models_dir, "mixed.wav")
    
    vad_model = os.path.join(models_dir, "silero_vad.onnx")
    campplus_model = os.path.join(models_dir, "campplus.onnx")
    
    print("\n=======================================================")
    print("STEP 1: Generating Mixed Multi-Speaker Audio File")
    print("=======================================================")
    audio, ground_truth = generate_mixed_audio(spk1_path, spk2_path, mixed_path, segment_len_sec=4.0)
    
    print("\n=======================================================")
    print("STEP 2: Initializing Local Speaker Diarizer Pipeline")
    print("=======================================================")
    start_init = time.time()
    diarizer = LocalSpeakerDiarizer(vad_model, campplus_model)
    init_time = time.time() - start_init
    print(f"Pipeline initialized in {init_time * 1000:.2f} ms")
    
    print("\n=======================================================")
    print("STEP 3: Running Diarization (VAD + Embedding + Cluster)")
    print("=======================================================")
    start_proc = time.time()
    
    results = diarizer.run(audio, n_clusters=2)
    
    total_time = time.time() - start_proc
    print(f"Diarization completed in {total_time * 1000:.2f} ms (Audio duration: {len(audio)/16000:.2f}s)")
    print(f"Real-time Factor (RTF): {total_time / (len(audio)/16000):.4f}x")
    
    print("\n=======================================================")
    print("STEP 4: Output Diarization Timeline")
    print("=======================================================")
    for res in results:
        print(f"  [{res['start']:.2f}s - {res['end']:.2f}s] Speaker {res['speaker']}")

if __name__ == "__main__":
    run_diarization_demo()
