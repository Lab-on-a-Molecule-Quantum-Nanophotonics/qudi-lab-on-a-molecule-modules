from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.module import LogicBase
from qudi.interface.simple_scanner_interface import SimpleScannerSettings, SimpleScannerConstraints

from qtpy import QtCore

class NILaserScanningSettingsLogic(LogicBase):
    scan_min = StatusVar("scan_min", -10)
    scan_max = StatusVar("scan_max", 10)
    sample_size = StatusVar("sample_size", 1000)
    exposure = StatusVar("exposure", 0.1)

    _ni_laser_spectrometer_interfuse = Connector(name="ni_laser_spectrometer_interfuse", interface="SimpleScannerInterface")

    def on_activate(self):
        self.ni_laser_spectrometer_interfuse = self._ni_laser_spectrometer_interfuse()
        self.update_settings()
    def on_deactivate(self):
        pass
    def set_setpoint(self, value):
        self.ni_laser_spectrometer_interfuse.set_setpoint(value)
    def get_setpoint(self):
        return self.ni_laser_spectrometer_interfuse.get_setpoint()
    def set_scan_min(self, value):
        self.scan_min = value
        self.update_settings()
    def set_scan_max(self, value):
        self.scan_max = value
        self.update_settings()
    def set_sample_size(self, value):
        self.sample_size = value
        self.update_settings()
    def get_exposure(self):
        return self.exposure 
    def set_exposure(self, value):
        self.exposure = value
        self.ni_laser_spectrometer_interfuse.exposure_time = value
    def scan_constraints(self):
        return (
            self.ni_laser_spectrometer_interfuse.constraints.scan_min, 
            self.ni_laser_spectrometer_interfuse.constraints.scan_max
        )
    def sample_size_constraints(self):
        return (
            self.ni_laser_spectrometer_interfuse.constraints.sample_rate_min, 
            self.ni_laser_spectrometer_interfuse.constraints.sample_rate_max
        )
    def update_settings(self):
        settings = SimpleScannerSettings(
            first_point = self.scan_min,
            last_point = self.scan_max,
            sample_rate = self.ni_laser_spectrometer_interfuse._sample_rate,
            number_of_samples = self.sample_size,
        )
        return self.ni_laser_spectrometer_interfuse.set_settings(settings)
        
    def set_continuous_reading(self):
        self.ni_laser_spectrometer_interfuse.reset_continuous()
        
