import numpy as np
import onnxruntime as ort

class SileroVAD:
    def __init__(self, model_path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.reset()
        
    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        
    def __call__(self, chunk_samples):
        if len(chunk_samples.shape) == 2:
            if chunk_samples.shape[0] == 1:
                chunk_samples = chunk_samples[0]
            elif chunk_samples.shape[1] == 1:
                chunk_samples = chunk_samples[:, 0]
        chunk_samples = np.expand_dims(chunk_samples, axis=0)
        x = np.concatenate([self._context, chunk_samples], axis=1).astype(np.float32)
        sr_tensor = np.array(16000, dtype=np.int64)
        outputs = self.session.run(
            ['output', 'stateN'],
            {
                'input': x,
                'state': self._state,
                'sr': sr_tensor
            }
        )
        prob = outputs[0][0, 0]
        self._state = outputs[1]
        self._context = x[:, -64:]
        return prob
