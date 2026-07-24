import os
import sys
import time
import argparse
import queue
import threading
import wave
import numpy as np
import json
import uuid
import ctypes
import socket
from collections import deque
import pyaudiowpatch as pyaudio
from diarizer import SileroVAD, CamPlusExtractor, SpeakerClusteringEngine
from diarizer.voice_profile_db import VoiceProfileDB, MAX_POOL_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
VAD_MODEL = os.path.join(MODELS_DIR, "silero_vad.onnx")
CAMPPLUS_MODEL = os.path.join(MODELS_DIR, "campplus.onnx")
LOG_FILE = os.path.join(os.path.dirname(__file__), "cli_diarizer.log")

class CommitSignalBuffer:
    def __init__(self, max_size: int = 300):
        self._ring = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._last_known_lc_state = 'UNKNOWN'
        self._last_heartbeat_ts = 0.0

    def push(self, signal: dict) -> None:
        if signal.get('type') == 'state':
            self._last_known_lc_state = signal.get('lc_state', 'UNKNOWN')
            self._last_heartbeat_ts = time.monotonic()
            return
        with self._lock:
            self._ring.append(signal)

    def get_window(self, start_sec: float, end_sec: float, epoch_offset_sec: float = 0.0, tolerance_sec: float = 0.2) -> list:
        with self._lock:
            snapshot = list(self._ring)
        result = []
        t_start = start_sec - tolerance_sec
        t_end = end_sec + tolerance_sec
        for item in snapshot:
            if 'acoustic_end_ms' in item:
                t = item['acoustic_end_ms'] / 1000.0 + epoch_offset_sec
                if t_start <= t <= t_end:
                    result.append(item)
        return result

    def get_state_at(self, tolerance_ms: float = 1500.0) -> str:
        age_ms = (time.monotonic() - self._last_heartbeat_ts) * 1000
        if age_ms > tolerance_ms:
            return 'UNKNOWN'
        return self._last_known_lc_state

class UdpSignalReceiver(threading.Thread):
    def __init__(self, buffer: CommitSignalBuffer, port: int = 47832):
        super().__init__(daemon=True)
        self._buf = buffer
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.bind(('127.0.0.1', port))
            self._sock.settimeout(1.0)
        except Exception as e:
            print(f"[UDP] Failed to bind to port {port}: {e}", file=sys.stderr)
            self._sock = None

    def run(self):
        if not self._sock:
            return
        print(f"[UDP] Listening for LiveCaption signals on port {self._port}...", file=sys.stderr)
        while True:
            try:
                data, _ = self._sock.recvfrom(512)
                signal = json.loads(data.decode('utf-8'))
                if signal.get('type') == 'commit':
                    print(f"[LiveCaption] Received commit signal: {signal.get('reason')} - {signal.get('acoustic_end_ms')} ms", file=sys.stderr)
                self._buf.push(signal)
            except socket.timeout:
                continue
            except Exception:
                pass

class CLIDiarizer:
    def __init__(self, device_idx, threshold=0.5, expected_speakers=2, min_speech_duration=1.2, db_path="models/voice_profiles.db", lc_port=47832, dump_wav=False, debug=False):
        self.device_idx = device_idx
        self.threshold = threshold
        self.expected_speakers = expected_speakers
        self.min_speech_duration = min_speech_duration
        self.dump_wav = dump_wav
        self.db_path = db_path
        self.lc_port = lc_port
        self.debug = debug
        
        self.sample_rate = 16000
        self.chunk_size = 512
        self.max_speech_duration = 15.0
        
        self.log_file = open(LOG_FILE, "a", encoding="utf-8")
        self.log("[INIT] Initializing CLI Diarizer...")
        
        try:
            self.vad = SileroVAD(VAD_MODEL)
            self.extractor = CamPlusExtractor(CAMPPLUS_MODEL)
        except Exception as e:
            self.emit({"type": "error", "message": f"Failed to load ONNX models: {e}"})
            raise e
            
        self.chunk_queue = queue.Queue()
        self.segment_queue = queue.Queue()
        
        self.is_running = False
        self.segment_registry = []
        self.clustering_engine = SpeakerClusteringEngine()
        
        self.rolling_buffer = np.zeros(0, dtype=np.float32)
        self.total_processed_samples = 0
        self.rolling_buffer_max_samples = 30 * self.sample_rate
        self.callback_resample_buffer = np.zeros(0, dtype=np.float32)
        
        self.p = None
        self.stream = None
        self.stream_samplerate = 16000
        self.stream_channels = 1
        
        self.vad_thread = None
        self.embedding_thread = None
        
        self.session_id = str(uuid.uuid4())
        self.db = VoiceProfileDB(self.db_path)
        self.uid_to_profile_id = {}
        self._session_embeddings = {}
        self._session_emb_ids = set()
        self._known_profiles = self.db.load_all_active()
        self.manual_merges = {}
        self._segs_since_recognition = {}
        self.speaker_profiles_data = {}
        
        self.lc_buffer = CommitSignalBuffer()
        self.lc_receiver = UdpSignalReceiver(self.lc_buffer, self.lc_port)
        self._lc_epoch_offset = 0.0
        self._recording_start_ticks = 0

    def emit(self, payload: dict):
        line = json.dumps(payload, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if self.debug and payload.get("type") != "vol_level":
            self.log(f"[DEBUG EMIT] {line}")

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_str = f"[{timestamp}] {message}"
        print(log_str, file=sys.stderr)
        if self.log_file:
            self.log_file.write(log_str + "\n")
            self.log_file.flush()

    def open_stream_safely(self, device_id, target_sr):
        time.sleep(0.1)
        self.p = pyaudio.PyAudio()
        try:
            dev_info = self.p.get_device_info_by_index(device_id)
            default_sr = int(dev_info['defaultSampleRate'])
            ch = int(loopback_channels) if (loopback_channels := dev_info.get('maxInputChannels', 1)) > 0 else 1
        except Exception as e:
            raise Exception(f"Failed to query device info: {e}")
            
        rate_candidates = [target_sr, default_sr, 48000, 44100, 16000]
        rate_candidates = list(dict.fromkeys([r for r in rate_candidates if r > 0]))
        
        def pyaudio_callback(in_data, frame_count, time_info, status):
            self.audio_callback(in_data, frame_count, time_info, status)
            return (None, pyaudio.paContinue)
            
        last_error = ""
        for sr in rate_candidates:
            blocksize = int(sr * 0.032)
            if blocksize <= 0:
                blocksize = 512
                
            try:
                stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=sr,
                    input=True,
                    input_device_index=device_id,
                    frames_per_buffer=blocksize,
                    stream_callback=pyaudio_callback
                )
                return stream, sr, ch
            except Exception as e:
                last_error = str(e)
                continue
                
        raise Exception(f"All combinations failed. Last error: {last_error}.")

    def audio_callback(self, in_data, frame_count, time_info, status):
        indata = np.frombuffer(in_data, dtype=np.float32)
        if self.stream_channels > 1:
            audio_mono = indata.reshape(-1, self.stream_channels)[:, 0]
        else:
            audio_mono = indata
            
        rms = float(np.sqrt(np.mean(audio_mono**2)))
        self.emit({"type": "vol_level", "rms": rms})
        
        if self.stream_samplerate != self.sample_rate:
            duration = len(audio_mono) / self.stream_samplerate
            num_target_samples = int(duration * self.sample_rate)
            x_orig = np.linspace(0, duration, len(audio_mono))
            x_target = np.linspace(0, duration, num_target_samples)
            audio_mono = np.interp(x_target, x_orig, audio_mono)
            
        if self.dump_wav and hasattr(self, 'wav_file') and self.wav_file:
            try:
                audio_int16 = (np.clip(audio_mono, -1.0, 1.0) * 32767.0).astype(np.int16)
                self.wav_file.writeframes(audio_int16.tobytes())
            except Exception:
                pass
            
        self.callback_resample_buffer = np.append(self.callback_resample_buffer, audio_mono)
        while len(self.callback_resample_buffer) >= self.chunk_size:
            chunk = self.callback_resample_buffer[:self.chunk_size].copy()
            self.callback_resample_buffer = self.callback_resample_buffer[self.chunk_size:]
            self.chunk_queue.put(chunk.reshape(1, -1))

    def run(self):
        self.is_running = True
        self.lc_receiver.start()
        self._recording_start_ticks = ctypes.windll.kernel32.GetTickCount64()
        self._lc_epoch_offset = -(self._recording_start_ticks / 1000.0)
        
        self.vad_thread = threading.Thread(target=self.vad_processing_loop, daemon=True)
        self.embedding_thread = threading.Thread(target=self.embedding_processing_loop, daemon=True)
        self.lc_state_thread = threading.Thread(target=self.lc_state_loop, daemon=True)
        
        self.vad_thread.start()
        self.embedding_thread.start()
        self.lc_state_thread.start()
        
        if self.dump_wav:
            try:
                self.wav_file = wave.open("cli_loopback_debug.wav", "wb")
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(2)
                self.wav_file.setframerate(self.sample_rate)
            except Exception as e:
                self.log(f"[WARNING] Failed to open debug WAV: {e}")
                self.wav_file = None
        
        try:
            self.stream, self.stream_samplerate, self.stream_channels = self.open_stream_safely(self.device_idx, self.sample_rate)
            self.stream.start_stream()
            self.emit({"type": "ready"})
        except Exception as e:
            self.emit({"type": "error", "message": f"Failed to open audio stream: {e}"})
            self.stop()
            return False

        self._command_loop()
        return True

    def lc_state_loop(self):
        last_state = None
        while self.is_running:
            state = self.lc_buffer.get_state_at()
            if state != last_state:
                self.emit({"type": "lc_state", "state": state})
                last_state = state
            time.sleep(0.1)

    def _command_loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            action = cmd.get("cmd")
            if action == "stop":
                break
            elif action == "set_config":
                self._handle_set_config(cmd)
            elif action == "label_speaker":
                self._handle_label_speaker(cmd)
            elif action == "merge_speakers":
                self._handle_merge_speakers(cmd)
            elif action == "reassign_segment":
                self._handle_reassign_segment(cmd)
        self.stop()

    def _handle_set_config(self, cmd):
        if "threshold" in cmd: self.threshold = cmd["threshold"]
        if "expected_speakers" in cmd: self.expected_speakers = cmd["expected_speakers"]
        if "min_speech_duration" in cmd: self.min_speech_duration = cmd["min_speech_duration"]
        self.log(f"Config updated: {cmd}")

    def _handle_label_speaker(self, cmd):
        uid = cmd.get("uid")
        name = cmd.get("display_name")
        if not uid or not name: return
        
        if uid not in self.uid_to_profile_id:
            if uid in self.speaker_profiles_data:
                centroid = self.speaker_profiles_data[uid]['centroid']
                embs = list(self._session_embeddings.get(uid, []))
                pid = self.db.create_profile(centroid, initial_embeddings=embs)
                self.uid_to_profile_id[uid] = pid
            else:
                return
        pid = self.uid_to_profile_id[uid]
        self.db.set_display_name(pid, name)
        self.db.set_metadata(pid, 'display_name', name)
        try:
            self.db.set_user_confirmed(pid, True)
        except AttributeError:
            pass

    def _handle_merge_speakers(self, cmd):
        uid1 = cmd.get("uid_source")
        uid2 = cmd.get("uid_target")
        if not uid1 or not uid2: return
        
        for uid in [uid1, uid2]:
            if uid not in self.uid_to_profile_id:
                if uid in self.speaker_profiles_data:
                    centroid = self.speaker_profiles_data[uid]['centroid']
                    embs = list(self._session_embeddings.get(uid, []))
                    pid = self.db.create_profile(centroid, initial_embeddings=embs)
                    self.uid_to_profile_id[uid] = pid
                    
        if uid1 in self.uid_to_profile_id and uid2 in self.uid_to_profile_id:
            pid1 = self.uid_to_profile_id[uid1]
            pid2 = self.uid_to_profile_id[uid2]
            self.db.merge_profiles(source_id=pid1, target_id=pid2, session_id=self.session_id)
            self.manual_merges[uid1] = uid2
            self.uid_to_profile_id[uid1] = pid2

    def _handle_reassign_segment(self, cmd):
        old_uid = cmd.get("old_uid")
        new_uid = cmd.get("new_uid")
        start_sec = cmd.get("start_sec")
        end_sec = cmd.get("end_sec")
        if not old_uid or not new_uid: return
        
        old_pid = self.uid_to_profile_id.get(old_uid)
        new_pid = self.uid_to_profile_id.get(new_uid)
        if not old_pid or not new_pid: return
        
        TOLERANCE = 0.5
        moved = []
        remaining = []
        old_embs = self._session_embeddings.get(old_uid, [])
        for item in old_embs:
            if abs(item['start'] - start_sec) < TOLERANCE and abs(item['end'] - end_sec) < TOLERANCE:
                item['session_id'] = self.session_id
                moved.append(item)
            else:
                remaining.append(item)
                
        if moved:
            self._session_embeddings[old_uid] = remaining
            if new_uid not in self._session_embeddings:
                self._session_embeddings[new_uid] = deque(maxlen=MAX_POOL_SIZE)
            self._session_embeddings[new_uid].extend(moved)
            for item in moved:
                self.db.add_embedding(new_pid, item['embedding'], session_id=item['session_id'], start_sec=item['start'], end_sec=item['end'])
            self.db._remove_embeddings_by_timestamp(old_pid, start_sec, end_sec, TOLERANCE)

    def vad_processing_loop(self):
        sr = self.sample_rate
        chunk_size = self.chunk_size
        
        is_speech = False
        start_chunk = 0
        silence_counter = 0
        chunk_index = 0
        
        while self.is_running or not self.chunk_queue.empty():
            try:
                chunk_data = self.chunk_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            min_speech_chunks = int(self.min_speech_duration * sr / chunk_size)
            min_silence_chunks = int(400 / (chunk_size / sr * 1000))
            
            chunk_flat = chunk_data.squeeze()
            self.rolling_buffer = np.append(self.rolling_buffer, chunk_flat)
            self.total_processed_samples += len(chunk_flat)
            
            if len(self.rolling_buffer) > self.rolling_buffer_max_samples:
                self.rolling_buffer = self.rolling_buffer[-self.rolling_buffer_max_samples:]
                
            prob = self.vad(chunk_data)
            
            if prob >= self.threshold:
                if not is_speech:
                    is_speech = True
                    start_chunk = chunk_index
                silence_counter = 0
            else:
                if is_speech:
                    silence_counter += 1
                    if silence_counter >= min_silence_chunks:
                        end_chunk = chunk_index - silence_counter + 1
                        duration_chunks = end_chunk - start_chunk
                        if duration_chunks >= min_speech_chunks:
                            self.extract_and_queue_segment(start_chunk, end_chunk)
                        is_speech = False
                        
            if is_speech:
                current_duration = (chunk_index - start_chunk) * chunk_size / sr
                if current_duration >= self.max_speech_duration:
                    end_chunk = chunk_index
                    self.extract_and_queue_segment(start_chunk, end_chunk)
                    start_chunk = chunk_index + 1
                    silence_counter = 0
                    
            chunk_index += 1

    def extract_and_queue_segment(self, start_chunk, end_chunk):
        sr = self.sample_rate
        chunk_size = self.chunk_size
        start_sample_abs = start_chunk * chunk_size
        end_sample_abs = end_chunk * chunk_size
        first_sample_abs = self.total_processed_samples - len(self.rolling_buffer)
        start_idx_rel = start_sample_abs - first_sample_abs
        end_idx_rel = end_sample_abs - first_sample_abs
        
        if start_idx_rel < 0: start_idx_rel = 0
        if end_idx_rel > len(self.rolling_buffer): end_idx_rel = len(self.rolling_buffer)
            
        segment_audio = self.rolling_buffer[start_idx_rel:end_idx_rel].copy()
        start_sec = start_sample_abs / sr
        end_sec = end_sample_abs / sr
        
        self.segment_queue.put({
            'start': start_sec,
            'end': end_sec,
            'audio': segment_audio
        })

    def _apply_livecaption_gate(self, seg_start: float, seg_end: float) -> str:
        signals = self.lc_buffer.get_window(seg_start, seg_end, self._lc_epoch_offset)
        if not signals:
            state = self.lc_buffer.get_state_at()
            if state == 'PENDING':
                return 'suppress'
            return 'allow'
        has_hard_commit = any(s.get('reason') == 'HardCommit' for s in signals)
        all_dangling = all(s.get('is_dangling') for s in signals)
        was_merged = any(s.get('was_merged') for s in signals)
        action = 'allow'
        if all_dangling:
            action = 'suppress'
        elif has_hard_commit:
            action = 'reinforce'
        elif was_merged:
            action = 'reinforce'
        return action

    def embedding_processing_loop(self):
        while self.is_running or not self.segment_queue.empty():
            try:
                seg_task = self.segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            start_sec = seg_task['start']
            end_sec = seg_task['end']
            audio_data = seg_task['audio']
            
            emb = self.extractor(audio_data)
            if np.allclose(emb, 0): continue
                
            self.segment_registry.append({
                'start': start_sec,
                'end': end_sec,
                'embedding': emb
            })
            if len(self.segment_registry) > 300:
                self.segment_registry.pop(0)
            
            self.run_clustering()
            
            for seg in self.segment_registry:
                uid = seg.get('uuid')
                emb_id = id(seg['embedding'])
                if uid and emb_id not in self._session_emb_ids:
                    if uid not in self._session_embeddings:
                        self._session_embeddings[uid] = deque(maxlen=MAX_POOL_SIZE)
                    self._session_embeddings[uid].append({
                        'embedding': seg['embedding'],
                        'session_id': self.session_id,
                        'start': seg['start'],
                        'end': seg['end']
                    })
                    self._session_emb_ids.add(emb_id)
                    if uid in self.uid_to_profile_id:
                        self.db.add_embedding(self.uid_to_profile_id[uid], seg['embedding'], session_id=self.session_id, start_sec=seg['start'], end_sec=seg['end'])
            
            MIN_SEGS_BEFORE_RECOGNIZE = 3
            REEVAL_INTERVAL = 10
            
            for uid, data in self.speaker_profiles_data.items():
                if uid in self.uid_to_profile_id:
                    segs_since_assign = self._segs_since_recognition.get(uid, 0) + 1
                    self._segs_since_recognition[uid] = segs_since_assign
                    if segs_since_assign < REEVAL_INTERVAL:
                        continue
                    self._segs_since_recognition[uid] = 0
                
                min_segs = MIN_SEGS_BEFORE_RECOGNIZE
                if uid in self._session_embeddings and self._session_embeddings[uid]:
                    first_seg = self._session_embeddings[uid][0]
                    t_start = first_seg['start'] - 0.5
                    t_end = first_seg['start'] + 0.5
                    signals = self.lc_buffer.get_window(t_start, t_end, self._lc_epoch_offset)
                    has_hard_or_merge = any(s.get('reason') == 'HardCommit' or s.get('was_merged') for s in signals)
                    if has_hard_or_merge:
                        min_segs = 1
                
                if data['count'] < min_segs:
                    continue
                    
                MIN_EMB_FOR_CONSISTENCY = 3
                INTRA_CONSISTENCY_RATIO = 0.50
                
                uid_pool = [item['embedding'] for item in self._session_embeddings.get(uid, [])]
                if len(uid_pool) >= MIN_EMB_FOR_CONSISTENCY:
                    new_pid, new_dist, consistency = self.db.recognize_from_pool(uid_pool, consistency_ratio=INTRA_CONSISTENCY_RATIO)
                    method = f"pool (cons={consistency:.2f})"
                else:
                    centroid = data['centroid']
                    new_pid, new_dist = self.db.recognize(centroid)
                    method = "centroid-fallback"
                
                if uid in self.uid_to_profile_id:
                    if new_pid and new_pid != self.uid_to_profile_id[uid]:
                        self.log(f"Reassign candidate for {uid}: was {self.uid_to_profile_id[uid]}, matches {new_pid} dist={new_dist:.3f}")
                else:
                    if new_pid:
                        self.uid_to_profile_id[uid] = new_pid
                        display = self.db.get_metadata(new_pid, 'display_name') or new_pid[:8]
                        self.log(f"Recognized {uid} -> {display}")
                        self.emit({
                            "type": "recognition",
                            "uid": uid,
                            "profile_id": new_pid,
                            "display_name": display,
                            "dist": float(new_dist),
                            "method": method
                        })
                        for item in self._session_embeddings[uid]:
                            self.db.add_embedding(new_pid, item['embedding'], session_id=item['session_id'], start_sec=item['start'], end_sec=item['end'])

    def run_clustering(self):
        if not self.segment_registry:
            return
            
        res = self.clustering_engine.process(self.segment_registry, self.expected_speakers, lc_gate_func=self._apply_livecaption_gate)
        self.segment_registry = res['segment_registry']
        self.speaker_profiles_data = res['speaker_profiles_data']
        
        processed_timeline = []
        for start, end, uid in res['timeline_lines']:
            if uid in self.manual_merges:
                uid = self.manual_merges[uid]
            processed_timeline.append((start, end, uid))
            
        segments = []
        TIMELINE_MIN_SEGS = 2
        for start, end, uid in processed_timeline:
            seg_count = self.speaker_profiles_data.get(uid, {}).get('count', 0)
            is_recognized = uid in self.uid_to_profile_id
            if seg_count < TIMELINE_MIN_SEGS and not is_recognized:
                continue
                
            identity_str = ""
            if uid in self.uid_to_profile_id:
                pid = self.uid_to_profile_id[uid]
                name = self.db.get_metadata(pid, 'display_name')
                if name: identity_str = name
                
            segments.append({
                "start": float(start),
                "end": float(end),
                "uid": uid,
                "identity": identity_str,
                "seg_count": seg_count
            })
            
        if segments:
            self.emit({
                "type": "timeline_update",
                "segments": segments,
                "speaker_count": len(self.speaker_profiles_data)
            })

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            
        if hasattr(self, 'wav_file') and self.wav_file:
            try:
                self.wav_file.close()
            except Exception:
                pass
            self.wav_file = None
            
        if self.p:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = None
            
        self._flush_session()
        self.emit({"type": "stopped"})
        if self.log_file:
            self.log_file.close()

    def _flush_session(self):
        try:
            for uid, data in self.speaker_profiles_data.items():
                if uid not in self.uid_to_profile_id:
                    embs = self._session_embeddings.get(uid, [])
                    pid = self.db.create_profile(data['centroid'], initial_embeddings=list(embs))
                    self.uid_to_profile_id[uid] = pid

            for uid, data in self.speaker_profiles_data.items():
                if data['count'] < 5:
                    embs = self._session_embeddings.get(uid, [])
                    if not embs: continue
                    is_all_dangling = True
                    for item in embs:
                        t_start = item['start'] - 0.2
                        t_end = item['end'] + 0.2
                        signals = self.lc_buffer.get_window(t_start, t_end, self._lc_epoch_offset)
                        if not signals or not all(s.get('is_dangling') for s in signals):
                            is_all_dangling = False
                            break
                    if is_all_dangling:
                        pid = self.uid_to_profile_id.get(uid)
                        if pid:
                            self.db.set_metadata(pid, "dangling", "true")

            self.db.flush_to_db(self.session_id)
            self.emit({"type": "session_flushed", "uid_map": self.uid_to_profile_id})
        except Exception as e:
            self.emit({"type": "error", "message": f"Flush error: {e}"})

def main():
    if "--list_devices" in sys.argv:
        p = pyaudio.PyAudio()
        devices = []
        try:
            device_count = p.get_device_count()
            for idx in range(device_count):
                dev = p.get_device_info_by_index(idx)
                host_api_info = p.get_host_api_info_by_index(dev['hostApi'])
                if dev['maxInputChannels'] > 0:
                    devices.append({
                        "index": idx,
                        "name": dev['name'],
                        "api": host_api_info['name'],
                        "is_loopback": dev.get('isLoopbackDevice', False)
                    })
        except Exception:
            pass
        finally:
            p.terminate()
        sys.stdout.write(json.dumps({"type": "device_list", "devices": devices}) + "\n")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Standalone CLI Diarizer Plugin")
    parser.add_argument("--device", type=int, default=0, help="Audio input device index")
    parser.add_argument("--db_path", type=str, default="models/voice_profiles.db", help="Path to voice profiles DB")
    parser.add_argument("--lc_port", type=int, default=47832, help="UDP port for LiveCaption sync")
    parser.add_argument("--dump_wav", action="store_true", help="Dump debug WAV file")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()
    
    diarizer = CLIDiarizer(
        device_idx=args.device,
        db_path=args.db_path,
        lc_port=args.lc_port,
        dump_wav=args.dump_wav,
        debug=args.debug
    )
    diarizer.run()

if __name__ == "__main__":
    main()
