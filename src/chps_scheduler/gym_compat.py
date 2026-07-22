from __future__ import annotations

from types import SimpleNamespace

import numpy as np


try:  # pragma: no cover - depends on optional RL installation
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # lightweight fallback for validation and deterministic demos
    class Env:
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = np.dtype(dtype)
            if shape is not None:
                self.low = np.full(shape, low, dtype=self.dtype)
                self.high = np.full(shape, high, dtype=self.dtype)
            else:
                self.low = np.asarray(low, dtype=self.dtype)
                self.high = np.asarray(high, dtype=self.dtype)
            self.shape = self.low.shape

        def sample(self):
            return np.random.default_rng().uniform(self.low, self.high).astype(self.dtype)

    gym = SimpleNamespace(Env=Env)
    spaces = SimpleNamespace(Box=Box)

