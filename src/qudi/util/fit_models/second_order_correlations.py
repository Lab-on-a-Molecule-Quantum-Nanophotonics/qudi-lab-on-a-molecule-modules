from qudi.util.fit_models.model import FitModelBase, estimator
import numpy as np

__all__ = ['ContinuousG2', 'continuous_g2_grandi2016']

def continuous_g2_grandi2016(gamma_1, gamma_2, S, tau):
    p = gamma_1 + gamma_2
    omega = np.sqrt(S * gamma_1 * gamma_2)
    q_square = (gamma_1-gamma_2)**2 - 4*omega**2
    q = np.sqrt(np.abs(q_square))
    if q_square < 0:
        q = 1j * q
    return np.real(1.0 - (p+q)/(2*q)*np.exp(-1/2 * (p-q) * np.abs(tau)) + (p-q)/(2*q)*np.exp(-1/2 * (p+q)*np.abs(tau)))

class ContinuousG2(FitModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_param_hint('gamma_1', value=1/3e-9, min=0.0, max=np.inf)
        self.set_param_hint('gamma_2', value=0.5/3e-9, min=0.0, max=np.inf)
        self.set_param_hint('S', value=1.0, min=0.0, max=np.inf)
        self.set_param_hint('g2_0', value=0.0, min=0.0, max=1.0)
        self.set_param_hint('t0', value=0.0, min=-np.inf, max=np.inf)
        self.set_param_hint('amplitude', value=1.0, min=0.0, max=np.inf)

    @staticmethod
    def _model_function(x, gamma_1, gamma_2, S, g2_0, t0, amplitude):
        return (continuous_g2_grandi2016(gamma_1, gamma_2, S, x-t0) - (1 - g2_0)) * amplitude

    @estimator("Dip")
    def estimate_dip(self, data, x):
        amplitude = np.median(data)
        i_min = np.argmin(data)
        g2_0 = data[i_min] / amplitude
        t0 = x[i_min]
        estimate = self.make_params()
        estimate['gamma_1'].set(value=1/3e-9, min=0.0, max=np.inf)
        estimate['gamma_2'].set(value=1/6e-9, min=0.0, max=np.inf)
        estimate['S'].set(value=0.0, min=0.0, max=np.inf)
        estimate['g2_0'].set(value=g2_0, min=0.0, max=1.0)
        estimate['t0'].set(value=t0, min=-np.inf, max=np.inf)
        estimate['amplitude'].set(value=amplitude, min=0, max=np.inf)
        print(f"estimated {estimate}")
        return estimate

