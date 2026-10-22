import time

import numpy as np
from PySide6 import QtCore
from qudi.core.configoption import ConfigOption

from qudi.interface.start_stop_interface import (
    InputConfiguration,
    InputEdge,
    StartStopHardwareState,
    StartStopHistogram,
    StartStopHistogramConfiguration,
    StartStopInterface,
)


class StartStopDummy(StartStopInterface):
    gamma_1 = ConfigOption(default=1/3.70e-9)
    gamma_2 = ConfigOption(default=0.5/3.70e-9)
    S = ConfigOption(default=1)
    photon_rate = ConfigOption(default=20e3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = StartStopHistogram(
                histogram=np.array([]),
                total_count=0,
                bin_width_ns=0.1,
                exposure_time_s=0,
                zero=0
                )
        self._configuration = StartStopHistogramConfiguration(
                enabled=True,
                bin_width_ns=0.1,
                histogram_length_ns=100.0,
                exposure_time_s=60.0
                )
        self._state = StartStopHardwareState.UNCONFIGURED
        self._start_time = time.perf_counter()
        self.config_A = InputConfiguration(0.5, InputEdge.RISING, 0.0)
        self.config_B = InputConfiguration(0.5, InputEdge.RISING, 0.0)

    def on_activate(self):
        self._state = StartStopHardwareState.UNCONFIGURED

    def on_deactivate(self):
        pass

    @property
    def inputs(self):
        return ["A", "B"]

    def configure_input(self, name:str, config: InputConfiguration):
        if name == "A":
            self.config_A = config
        elif name == "B":
            self.config_B = config
        else:
            raise ValueError(f"Input '{name}' does not exist.")

    def input_configuration(self, name:str):
        if name == "A":
            return self.config_A 
        elif name == "B":
            return self.config_B
        else:
            raise ValueError(f"Input '{name}' does not exist.")

    @property
    def histograms(self):
        return ["g2"]

    def histogram_configuration(self, histogram: str) -> StartStopHistogramConfiguration:
        """Return the current configuration of the given `histogram`."""
        if histogram == "g2":
            return self._configuration
        else:
            raise ValueError(f"Channel {histogram} does not exist.")

    def configure_histogram(self, histogram: str, config: StartStopHistogramConfiguration) -> None:
        """Configure the given `histogram`."""
        if histogram == "g2":
            self._configuration = config
        else:
            raise ValueError(f"Channel {histogram} does not exist.")

    @property
    def status(self) -> StartStopHardwareState:
        """Return the current status of the hardware."""
        return self._state

    def start_acquisition(self) -> StartStopHardwareState:
        """Start acquisition on all enabled histogram histograms."""
        self.log.debug("Dummy start stop started acquisition!")
        self._state = StartStopHardwareState.RUNNING
        self._start_time = time.perf_counter()
        bin_count = int(self._configuration.histogram_length_ns / self._configuration.bin_width_ns)
        self._data = StartStopHistogram(
                zero = bin_count//2,
                histogram=np.zeros(bin_count, dtype=np.int32),
                total_count=0,
                bin_width_ns=self._configuration.bin_width_ns,
                exposure_time_s=0,
                )
        QtCore.QTimer.singleShot(int(self._configuration.exposure_time_s*1000), self.stop_acquisition)
        return self._state

    def stop_acquisition(self) -> StartStopHardwareState:
        """Stop acquisition on all enabled histogram histograms."""
        self.log.debug("Dummy start stop stopped acquisition!")
        self._state = StartStopHardwareState.IDLE
        return self._state

    def pause_acquisition(self) -> StartStopHardwareState:
        """Pause acquisition on all enabled histogram histograms."""
        self.log.debug("Dummy start stop paused acquisition!")
        self._state = StartStopHardwareState.PAUSED
        return self._state

    def continue_acquisition(self) -> StartStopHardwareState:
        """Continue acquisition on all enabled histogram histograms."""
        self.log.debug("Dummy start stop restarted acquisition!")
        self._state = StartStopHardwareState.RUNNING
        return self._state

    def histogram_data(self, histogram: str) -> StartStopHistogram:
        """Get the current data for histogram `histogram`."""
        if histogram == "g2":
            return self.__get_or_generate_data()
        else:
            raise ValueError(f"Channel {histogram} does not exist.")

    def _g2(self, tau):
        p = self.gamma_1 + self.gamma_2
        omega = np.sqrt(self.S*self.gamma_1*self.gamma_2)
        q_square = (self.gamma_1 - self.gamma_2)**2 - 4*omega**2
        q = np.sqrt(np.abs(q_square))
        if q_square < 0:
            q = 1j * q
        return np.real(1 - (p+q)/(2*q)*np.exp(-1/2 * (p-q) * np.abs(tau)) + (p-q)/(2*q)*np.exp(-1/2 * (p+q)*np.abs(tau)))

    def __get_or_generate_data(self):
        if self._state is not StartStopHardwareState.RUNNING:
            return self._data
        now = time.perf_counter()
        ellapsed_time = now - self._start_time
        number_of_photons = int(self.photon_rate * ellapsed_time / 1000) # too lazy to do  a realistic calculation
        bin_count = int(self._configuration.histogram_length_ns / self._configuration.bin_width_ns)
        histgram_edges = np.linspace(
                start=-self._configuration.histogram_length_ns/2,
                stop=-self._configuration.histogram_length_ns/2,
                num = bin_count
                )
        ideal_g2 = self._g2(histgram_edges*1e-9)
        normalized_g2 = ideal_g2 / ideal_g2.sum()
        cdf = np.cumsum(normalized_g2)
        proba_values = np.random.uniform(
                low=0, 
                high=1, 
                size=number_of_photons
        )
        for p in proba_values:
            cdf_greater = np.nonzero(cdf >= p)
            if len(cdf_greater) <= 0:
                self._data.histogram[-1] += 1
            else:
                self._data.histogram[cdf_greater[0]] += 1
        self._start_time = now
        return self._data



