from typing import Iterable, Mapping, Union, Optional, Tuple, Type, Dict

import numpy as np

from qudi.interface.spectrometer_interface import SpectrometerInterface
from qudi.interface.simple_scanner_interface import SimpleScannerInterface, SimpleScannerSettings, SimpleScannerConstraints
from qudi.interface.finite_sampling_io_interface import SamplingOutputMode
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption

class NIMOGLabsSpectrometerInterfuse(SpectrometerInterface, SimpleScannerInterface):
    _ni_finite_sampling_io = Connector(name='scan_hardware', interface='FiniteSamplingIOInterface')
    _ni_setpoint = Connector(name="setpoint_hardware", interface="ProcessSetpointInterface")
    _moglabs_laser = Connector(name='wavelength_hardware', interface='FiniteSamplingInputInterface')
    
    _ni_output_channel = ConfigOption(name='analog_output_channel', missing='error')
    _ni_input_channel = ConfigOption(name='counter_input_channel', missing='error')
    
    _sample_rate = StatusVar(name="sample_rate", default=1e3)
    _number_of_samples = StatusVar(name="number_of_samples", default=2**12)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_activate(self):
        ni_io = self._ni_finite_sampling_io()
        self._constraints = SimpleScannerConstraints(
            scan_range = ni_io.constraints.output_channel_limits[self._ni_output_channel],
            sample_rate_range = (ni_io.constraints.min_sample_rate, ni_io.constraints.max_sample_rate),
            number_of_samples_range = (ni_io.constraints.min_frame_size, ni_io.constraints.max_frame_size),
        )
        self._settings = SimpleScannerSettings(
            first_point = self._constraints.scan_min,
            last_point = self._constraints.scan_max,
            sample_rate = self._sample_rate,
            number_of_samples = self._number_of_samples,
        )
    def on_deactivate(self):
        self._sample_rate = self._settings.sample_rate
        self._number_of_samples = self._settings.number_of_samples

    # Spectrometer interface
    def record_spectrum(self):
        wavelength,signal = self.get_frame()
        roi = wavelength != 0
        return np.vstack((wavelength[roi], signal[roi]))
    @property
    def exposure_time(self):
        return 1 / self._sample_rate
    @exposure_time.setter
    def exposure_time(self, value):
        """ Set the acquisition exposure time

        @param (float) value: Exposure time to set in second
        """
        self.settings.sample_rate = 1/value
        
    # SimpleScannerInterface
    @property
    def constraints(self):
        return self._constraints
    @property
    def settings(self):
        return self._settings
    def set_settings(self, settings: SimpleScannerSettings) -> Tuple[bool, Iterable[str]]:
        success, errors = settings.is_compatible_with(self.constraints)
        if success:
            self._settings = settings
            return True,[]
        else:
            return False, errors
    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        self.log.info("getting frame.")
        self._moglabs_laser().set_sample_rate(self.settings.sample_rate)
        self._moglabs_laser().set_active_channels(['wavelength'])
        self._moglabs_laser().set_frame_size(self.settings.number_of_samples)
        self._ni_finite_sampling_io().set_sample_rate(self.settings.sample_rate)
        self._ni_finite_sampling_io().set_output_mode(SamplingOutputMode.EQUIDISTANT_SWEEP)
        self._ni_finite_sampling_io().set_active_channels([self._ni_input_channel], [self._ni_output_channel])
        self._moglabs_laser().start_buffered_acquisition()
        self.log.info("waiting for acauisition to proceed.")
        d_ni_card = self._ni_finite_sampling_io().get_frame({self._ni_output_channel:(self.settings.first_point, self.settings.last_point, self.settings.number_of_samples)})
        d_wavemeter = self._moglabs_laser().get_buffered_samples()
        self._moglabs_laser().stop_buffered_acquisition()
        wavelength_perm = np.argsort(d_wavemeter["wavelength"])
        self.log.info("returning to manual value")
        self.set_setpoint(self.get_setpoint())
        return d_wavemeter["wavelength"][wavelength_perm], d_ni_card[self._ni_input_channel][wavelength_perm]
    def get_setpoint(self) -> float:
        self._ni_setpoint().set_activity_state(self._ni_output_channel, True)
        return self._ni_setpoint().get_setpoint(self._ni_output_channel)
    def set_setpoint(self, value: float) -> None:
        self._ni_setpoint().set_activity_state(self._ni_output_channel, True)
        self._ni_setpoint().set_setpoint(self._ni_output_channel, value)
    def reset_continuous(self):
        self._moglabs_laser().stop_buffered_acquisition()
        self._ni_finite_sampling_io().stop_buffered_acquisition()
    
    # Other
    @property
    def scan_step(self):
        return (self.settings.last_point - self.settings.first_point) / self.settings.number_of_samples
