import pathlib
import pickle
import numpy as np
from qudi.util.mutex import Mutex

class PhysicalModel(object):
    def __new__(cls, f0=None, A=None, sigma=None, bg=None, lag=None, idle=None):
        if not hasattr(cls, 'instance'):
            path = (pathlib.Path(__file__).parent / "data.pickle")
            if path.is_file():
                with open(path.resolve(), 'rb') as f:
                    data = pickle.load(f)
            else:
                rng = np.random.default_rng()
                data = dict(
                    f0 = rng.normal(0, 10e9) if f0 is None else f0,
                    A = rng.lognormal(0, 1) * 5e3 if A is None else A,
                    sigma = rng.lognormal(0, 1) * 50e6 if sigma is None else sigma,
                    bg = rng.lognormal(0, 1) if bg is None else bg,
                    lag = rng.lognormal(0, 1) * 100e6 if lag is None else lag
                )
                with open(path.resolve(), 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            cls.instance = super(PhysicalModel, cls).__new__(cls)
            cls.instance.f0 = data['f0']
            cls.instance.A = data['A']
            cls.instance.sigma = data['sigma']
            cls.instance.bg = data['bg']
            cls.instance.lag = data['lag']
        return cls.instance

    def __init__(self, idle=None):
        print("Inited!", locals())
        self._idle = 0.0 if idle is None else idle
        self.rng = np.random.default_rng()
        self._lock = Mutex()

    def __call__(self, f=None):
        _f = self.idle + self.lag if f is None else f
        return self.rng.poisson(self.A * self.sigma**2 / (self.sigma**2 + (_f - self.f0)**2) + self.bg)

    @property 
    def idle(self):
        with self._lock:
            return self._idle

    @idle.setter
    def idle(self, v):
        with self._lock:
            self._idle = v

    def as_dict(self):
        return dict(f0=self.f0, A=self.A, sigma=self.sigma, bg=self.bg, lag=self.lag, idle=self.idle)

