import sqlite3
import uuid
import datetime
from contextlib import closing
import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import threading
from dataclasses import dataclass

MAX_POOL_SIZE          = 75
COARSE_GATE            = 0.35
FINE_GATE              = 0.30
MIN_VOTE_RATIO         = 0.25
MIN_VOTE_ABS           = 3
POOL_PURITY_THRESHOLD  = 0.15
MIN_CONFIRMED_SEGS     = 5
TEMPORAL_SPREAD_MIN    = 60.0

@dataclass
class ProfileCache:
    profile_id:   str
    display_name: str
    centroid:     np.ndarray
    pool:         List[np.ndarray]
    pool_meta:    List[dict]
    pool_matrix:  Optional[np.ndarray]
    segment_count: int
    dirty:        bool
    is_user_confirmed: bool
    temporal_spread: float

class VoiceProfileDB:
    CROSS_SESSION_RECOGNITION_THRESH = 0.30

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache: Dict[str, ProfileCache] = {}
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        if self.db_path != ':memory:':
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    profile_id    TEXT PRIMARY KEY,
                    display_name  TEXT NOT NULL DEFAULT '',
                    centroid_blob BLOB NOT NULL,
                    created_at    TEXT NOT NULL,
                    last_seen_at  TEXT NOT NULL,
                    session_count INTEGER NOT NULL DEFAULT 1,
                    segment_count INTEGER NOT NULL DEFAULT 0,
                    is_active     INTEGER NOT NULL DEFAULT 1,
                    is_user_confirmed INTEGER NOT NULL DEFAULT 0
                )
            ''')
            conn.commit()

    def load_all_active(self) -> list:
        return []

    def recognize(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        return None, 1.0

    def recognize_from_pool(self, query_embeddings: List[np.ndarray], consistency_ratio: float = 0.50) -> Tuple[Optional[str], float, float]:
        return None, 1.0, 0.0

    def create_profile(self, centroid: np.ndarray, initial_embeddings: List[dict] = None, display_name: str = '') -> str:
        return str(uuid.uuid4())

    def add_embedding(self, profile_id: str, embedding: np.ndarray, session_id: str = '', start_sec: float = 0.0, end_sec: float = 0.0) -> None:
        pass

    def flush_to_db(self, session_id: str) -> None:
        pass

    def set_display_name(self, profile_id: str, name: str) -> None:
        pass

    def set_metadata(self, profile_id: str, key: str, value: str) -> None:
        pass

    def get_metadata(self, profile_id: str, key: str) -> str:
        return ""
