from qudi.interface.spectrometer_interface import SpectrometerInterface
from qudi.interface.finite_sampling_io_interface import SamplingOutputMode
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption

class NIMOGLabsSpectrometerInterfuse(SpectrometerInterface):
    _ni_finite_sampling_io = Connector(name='scan_hardware', interface='FiniteSamplingIOInterface')
    _moglabs_laser = Connector(name='wavelength_hardware', interface='FiniteSamplingInputInterface')
    _ni_output_channel = ConfigOption(name='analog_output_channel', missing='error')
    _ni_input_channel = ConfigOption(name='counter_input_channel', missing='error')

    def on_activate(self):
        self._scan_min = -2 #V
        self._scan_max = 2 #V
        self._sample_rate = 100 #Hz
        self._sample_size = 1000

    def record_spectrum(self):
        self._moglabs_laser.set_sample_rate(self.sample_rate)
        self._moglabs_laser.set_active_channels(['wavelength'])
        self._moglabs_laser.set_frame_size(self.frame_size)
        self._ni_finite_sampling_io.set_sample_rate(self.sample_rate)
        self._ni_finite_sampling_io.set_frame_size(self.frame_size)
        self._ni_finite_sampling_io.set_output_mode(SamplingOutputMode.EQUIDISTANT_SWEEP)
        self._ni_finite_sampling_io.set_active_channels([self._ni_input_channel], [self._ni_output_channel])
        self._moglabs_laser.start_buffered_acquisition()
        self._ni_finite_sampling_io.get_frame({self._ni_output_channel:(self._scan_min, self._scan_max, self.scan_step)})
        return super().record_spectrum()
    @property
    def exposure_time(self):
        return 1 / self._sample_rate
    @exposure_time.setter
    def exposure_time(self, value):
        """ Set the acquisition exposure time

        @param (float) value: Exposure time to set in second
        """
        self._sample_rate = 1/value

    @property
    def scan_step(self):
        return (self._scan_max - self._scan_min) / self._sample_size

    @property
    def scan_min(self):
        return self._scan_min

    def set_scan_min(self, v):
        self._scan_min = v

    @property
    def scan_max(self):
        return self._scan_max

    def set_scan_max(self, v):
        self._scan_max = v

    @property
    def sample_size(self):
        return self._sample_size

    def set_sample_size(self, v):
        self._sample_size = v
