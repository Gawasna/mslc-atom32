import numpy as np
from sklearn.cluster import SpectralClustering
from .vad import SileroVAD
from .embedding import CamPlusExtractor

class LocalSpeakerDiarizer:
    def __init__(self, vad_model_path, campplus_model_path):
        self.vad = SileroVAD(vad_model_path)
        self.extractor = CamPlusExtractor(campplus_model_path)
        
    def segment_speech(self, audio, threshold=0.5, min_speech_ms=250, min_silence_ms=300):
        self.vad.reset()
        chunk_size = 512
        sr = 16000
        min_speech_chunks = int(min_speech_ms / (chunk_size / sr * 1000))
        min_silence_chunks = int(min_silence_ms / (chunk_size / sr * 1000))
        num_samples = len(audio)
        num_chunks = num_samples // chunk_size
        segments = []
        is_speech = False
        start_chunk = 0
        silence_counter = 0
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = audio[start_idx:end_idx]
            prob = self.vad(chunk)
            if prob >= threshold:
                if not is_speech:
                    is_speech = True
                    start_chunk = i
                silence_counter = 0
            else:
                if is_speech:
                    silence_counter += 1
                    if silence_counter >= min_silence_chunks:
                        end_chunk = i - silence_counter + 1
                        duration_chunks = end_chunk - start_chunk
                        if duration_chunks >= min_speech_chunks:
                            segments.append((start_chunk * chunk_size, end_chunk * chunk_size))
                        is_speech = False
        if is_speech:
            duration_chunks = num_chunks - start_chunk
            if duration_chunks >= min_speech_chunks:
                segments.append((start_chunk * chunk_size, num_chunks * chunk_size))
        sec_segments = [(start / sr, end / sr) for start, end in segments]
        return sec_segments

    def run(self, audio, n_clusters=2, threshold=0.5):
        segments = self.segment_speech(audio, threshold=threshold)
        if not segments:
            print("No speech detected.")
            return []
        print(f"Detected {len(segments)} speech segments.")
        embeddings = []
        valid_segments = []
        sr = 16000
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_audio = audio[start_sample:end_sample]
            emb = self.extractor(segment_audio)
            if not np.allclose(emb, 0):
                embeddings.append(emb)
                valid_segments.append((start_sec, end_sec))
        if not embeddings:
            return []
        embeddings = np.array(embeddings)
        if len(valid_segments) == 1 or n_clusters == 1:
            labels = np.zeros(len(valid_segments), dtype=int)
        else:
            n_clusters = min(n_clusters, len(valid_segments))
            similarity_matrix = np.dot(embeddings, embeddings.T)
            affinity_matrix = (similarity_matrix + 1.0) / 2.0
            sc = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                random_state=42
            )
            labels = sc.fit_predict(affinity_matrix)
        results = []
        for (start_sec, end_sec), label in zip(valid_segments, labels):
            results.append({
                'start': round(start_sec, 2),
                'end': round(end_sec, 2),
                'speaker': int(label)
            })
        return results
