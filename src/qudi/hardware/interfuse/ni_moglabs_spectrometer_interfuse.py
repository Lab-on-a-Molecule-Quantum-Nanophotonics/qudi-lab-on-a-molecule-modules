from typing import Iterable, Mapping, Union, Optional, Tuple, Type, Dict

import time
import numpy as np

from qudi.interface.spectrometer_interface import SpectrometerInterface
from qudi.interface.simple_scanner_interface import SimpleScannerInterface, SimpleScannerSettings, SimpleScannerConstraints
from qudi.interface.finite_sampling_io_interface import SamplingOutputMode
from qudi.interface.process_control_interface import ProcessControlConstraints, ProcessControlInterface
from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption
from qudi.util.overload import OverloadedAttribute

from qtpy import QtCore


_Real = Union[int, float]

class NIMOGLabsSpectrometerInterfuse(SpectrometerInterface, SimpleScannerInterface, ProcessControlInterface):
    _ni_finite_sampling_io = Connector(name='scan_hardware', interface='FiniteSamplingIOInterface')
    _ni_setpoint = Connector(name="setpoint_hardware", interface="ProcessSetpointInterface")
    _moglabs_laser = Connector(name='wavelength_hardware', interface='FiniteSamplingInputInterface')
    _moglabs_grating = Connector(name='grating', interface='MotorInterface')
    
    _ni_output_channel = ConfigOption(name='analog_output_channel', missing='error')
    _ni_input_channel = ConfigOption(name='counter_input_channel', missing='error')
    _grating_axis = ConfigOption(name='grating_axis', default='grating')
    _mini_tension = ConfigOption(name='tension_min', default=-2.5)
    _maxi_tension = ConfigOption(name='tension_max', default=2.5)
    
    _sample_rate = StatusVar(name="sample_rate", default=1e3)
    _number_of_samples = StatusVar(name="number_of_samples", default=2**12)
    _wavelengths = StatusVar(name='frequencies', default=np.zeros(0))
    _grating_positions = StatusVar(name='grating_positions', default=np.zeros(0))
    
    sigCalibrationChanged = QtCore.Signal()
    
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
        if len(self._wavelengths) <= 0:
            self.calibrate()
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
    constraints = OverloadedAttribute()
    @constraints.overload("SimpleScannerInterface")
    @property
    def constraints(self):
        return self._constraints
    @property
    def settings(self):
        return self._settings
    def set_settings(self, settings: SimpleScannerSettings) -> Tuple[bool, Iterable[str]]:
        success, errors = settings.is_compatible_with(self._constraints)
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
    def reset_continuous(self):
        self._moglabs_laser().stop_buffered_acquisition()
        self._ni_finite_sampling_io().stop_buffered_acquisition()
    
    @QtCore.Slot()
    def calibrate(self):    
        mod_status = self.set_setpoint("tension", 0)
        constraints = self._moglabs_grating().get_constraints()[self._grating_axis]
        self._motor_positions = np.arange(start=constraints['pos_min'], stop=constraints['pos_max'], step=constraints['pos_step'])
        self._wavelengths = np.empty(len(self._motor_positions), dtype=np.float64)
        self.log.info("Starting calibration scan...")
        self._moglabs_laser().start_stream()
        for i in range(len(self._motor_positions)):
            self._moglabs_grating().move_abs({self._grating_axis: self._motor_positions[i]})
            while self._moglabs_grating().get_pos()[self._grating_axis] != self._motor_positions[i]:
                time.sleep(0.1)
            self._wavelengths[i] = self._moglabs_laser().read_data(10)[0].mean()
        self._moglabs_laser().stop_stream()
        self.log.info("Calibration scan done.")
        self.sigCalibrationChanged.emit()
        
    def get_calibration(self):
        return self._grating_positions, self._wavelengths
    
    # Other
    @property
    def scan_step(self):
        return (self.settings.last_point - self.settings.first_point) / self.settings.number_of_samples

    # ProcessControlInterface
    @constraints.overload("ProcessControlInterface")
    @property
    def constraints(self):
        return ProcessControlConstraints(
            setpoint_channels = ["grating wavelength", "tension", "grating position setpoint"],
            process_channels = ["wavelength", "grating position"],
            units = {
                "grating wavelength": "nm",
                "tension": "V",
                "grating position setpoint": "step",
                "wavelength": "nm",
                "grating position": "step"
            },
            limits = {
                "grating wavelength": (min(self._wavelengths), max(self._wavelengths)),
                "tension": (self._mini_tension, self._maxi_tension),
                "grating position setpoint": (min(self._grating_positions), max(self._grating_positions)),
                "wavelength": (400.0,2000.0),
                "grating position": (min(self._grating_positions), max(self._grating_positions))
            },
            dtypes = {
                "grating wavelength": np.float64,
                "tension": np.float64,
                "grating position setpoint": int,
                "wavelength": np.float64,
                "grating position": int
            }
        )
        
    def set_activity_state(self, channel: str, active: bool) -> None:
        pass
        
    def get_activity_state(self, channel: str) -> bool:
        return True
        
    def set_setpoint(self, channel: str, value: _Real) -> None:
        if channel == "grating wavelength":
            self.set_setpoint(
                "grating position setpoint", 
                self._motor_positions[np.argmin(np.abs(self._wavelengths-value))][0]
            )
        elif channel == "tension":
            self._ni_setpoint().set_activity_state(self._ni_output_channel, True)
            self._ni_setpoint().set_setpoint(self._ni_output_channel, value)
        elif channel == "grating position setpoint":
            self._moglabs_grating().move_abs({self._grating_axis: value})
        else:
            raise ValueError("No channel named %s" % channel)
        
    def get_setpoint(self, channel: str):
        if channel == "grating wavelength":
            return self._wavelengths[np.argmin(np.abs(self._motor_positions-self.get_setpoint("grating position setpoint")))][0]
        elif channel == "tension":
            if self._ni_setpoint().get_activity_state(self._ni_output_channel):
                return self._ni_setpoint().get_setpoint(self._ni_output_channel)
            else:
                return 0.0
        elif channel == "grating position setpoint":
            return self._moglabs_grating().get_pos()[self._grating_axis]
        else:
            raise ValueError("No channel named %s" % channel)
            
    def get_process_value(self, channel: str) -> _Real:
        """ Get current process value for a single channel """
        if channel == "wavelength":
            self._moglabs_laser().start_stream()
            v = self._moglabs_laser().read_data(10)[0].mean()
            self._moglabs_laser().stop_stream()
            return v
        elif channel == "grating position":
            return self._moglabs_grating().get_pos()[self._grating_axis]
        else:
            raise ValueError("No channel named %s" % channel)