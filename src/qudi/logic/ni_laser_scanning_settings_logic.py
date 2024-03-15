from qudi.core.connector import Connector
from qudi.core.statusvariable import StatusVar
from qudi.core.module import LogicBase

from qtpy import QtCore

class NILaserScanningSettingsLogic(LogicBase):
    scan_min = StatusVar("scan_min", -10)
    scan_max = StatusVar("scan_max", -10)
    sample_size = StatusVar("sample_size", 1000)

    ni_laser_spectrometer_interfuse = Connector("NIMOGLabsSpectrometerInterfuse")

    def on_activate(self):
        self.ni_laser_spectrometer_interfuse.set_scan_min(self.scan_min)
        self.ni_laser_spectrometer_interfuse.set_scan_max(self.scan_max)
        self.ni_laser_spectrometer_interfuse.set_sample_size(self.sample_size)
    def on_deactivate(self):
        pass
    def set_scan_min(self, value):
        self.scan_min = value
        self.ni_laser_spectrometer_interfuse.set_scan_min(value)
    def set_scan_max(self, value):
        self.scan_max = value
        self.ni_laser_spectrometer_interfuse.set_scan_max(value)
    def set_sample_size(self, value):
        self.sample_size = value
        self.ni_laser_spectrometer_interfuse.set_sample_size(value)
