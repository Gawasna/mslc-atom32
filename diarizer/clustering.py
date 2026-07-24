import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

class SpeakerClusteringEngine:
    K_REDUCTION_MARGIN = 0.04
    SPLIT_PREFERENCE_MARGIN = 0.02

    def __init__(self):
        self.persistent_id_counter = 1
        self.speaker_colors = ["#4A90E2", "#F5A623", "#7ED321", "#BD10E0", "#9013FE", "#50E3C2"]
        self.established_k = 1
        self._singleton_age: dict = {}

    def process(self, segment_registry, expected_speakers, lc_gate_func=None):
        n_segments = len(segment_registry)
        embeddings = np.array([seg['embedding'] for seg in segment_registry])
        
        if n_segments == 1 or expected_speakers == 1:
            labels = np.zeros(n_segments, dtype=int)
        else:
            similarity = np.dot(embeddings, embeddings.T)
            similarity = np.clip(similarity, -1.0, 1.0)
            cosine_distance = 1.0 - similarity
            
            if lc_gate_func:
                for i in range(n_segments - 1):
                    seg_end = segment_registry[i]['end']
                    next_start = segment_registry[i+1]['start']
                    action = lc_gate_func(seg_end, next_start)
                    if action == 'suppress':
                        cosine_distance[i, i+1] = 0.0
                        cosine_distance[i+1, i] = 0.0
                    elif action == 'reinforce':
                        cosine_distance[i, i+1] = 1.0
                        cosine_distance[i+1, i] = 1.0
            
            try:
                if expected_speakers > 1:
                    n_clust = min(expected_speakers, n_segments)
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clust,
                        metric='precomputed',
                        linkage='average'
                    )
                    labels = clustering.fit_predict(cosine_distance)
                else:
                    max_dist = np.max(cosine_distance)
                    
                    if max_dist < 0.35:
                        labels = np.zeros(n_segments, dtype=int)
                    else:
                        best_k = 2
                        best_score = -1.0
                        best_labels = None
                        
                        max_k = min(n_segments - 1, 6)
                        if max_k < 2:
                            max_k = 2
                            
                        if n_segments == 2:
                            if max_dist > 0.25:
                                labels = np.array([0, 1])
                            else:
                                labels = np.zeros(n_segments, dtype=int)
                        else:
                            all_scores = {}
                            for k in range(2, max_k + 1):
                                clustering = AgglomerativeClustering(
                                    n_clusters=k,
                                    metric='precomputed',
                                    linkage='average'
                                )
                                tmp_labels = clustering.fit_predict(cosine_distance)
                                try:
                                    score = silhouette_score(cosine_distance, tmp_labels, metric='precomputed')
                                except ValueError:
                                    score = -1.0
                                all_scores[k] = (score, tmp_labels)

                            sorted_ks = sorted(all_scores.keys(), reverse=True)
                            chosen_k = sorted_ks[0]
                            chosen_score = all_scores[chosen_k][0]
                            for k in sorted_ks[1:]:
                                s = all_scores[k][0]
                                if s > chosen_score + self.SPLIT_PREFERENCE_MARGIN:
                                    chosen_k = k
                                    chosen_score = s

                            effective_established_k = min(self.established_k, max_k)
                            if chosen_k < effective_established_k and effective_established_k in all_scores:
                                established_score = all_scores[effective_established_k][0]
                                gain = chosen_score - established_score
                                if gain < self.K_REDUCTION_MARGIN:
                                    chosen_k = effective_established_k
                                    chosen_score = established_score

                            best_k = chosen_k
                            best_score = all_scores[best_k][0]
                            best_labels = all_scores[best_k][1]

                            raw_best_k = max(all_scores, key=lambda k: all_scores[k][0])
                            raw_best_score = all_scores[raw_best_k][0]

                            if raw_best_score > 0.15:
                                prev_k = raw_best_k - 1
                                if prev_k in all_scores:
                                    prev_score = all_scores[prev_k][0]
                                    if raw_best_score - prev_score >= self.SPLIT_PREFERENCE_MARGIN:
                                        self.established_k = max(self.established_k, raw_best_k)
                                else:
                                    self.established_k = max(self.established_k, raw_best_k)

                            if best_labels is not None:
                                labels = best_labels
                            else:
                                labels = np.zeros(n_segments, dtype=int)
            except Exception as e:
                labels = np.zeros(n_segments, dtype=int)
                
        for i, c in enumerate(labels):
            segment_registry[i]['uuid'] = f"Speaker-{c+1:02d}"

        return {
            'segment_registry': segment_registry,
            'speaker_profiles_data': {},
            'timeline_lines': [(s['start'], s['end'], s['uuid']) for s in segment_registry]
        }
