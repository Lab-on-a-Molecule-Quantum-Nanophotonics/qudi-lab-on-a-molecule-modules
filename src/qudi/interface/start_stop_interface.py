from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from qudi.core.module import Base


class InputEdge(Enum):
    RISING = 0
    FALLING = 1


@dataclass
class InputConfiguration:
    level: float
    edge: InputEdge
    delay: float


@dataclass
class StartStopHistogramConfiguration:
    enabled: bool
    bin_width_ns: float
    histogram_length_ns: float
    exposure_time_s: float


class StartStopHardwareState(Enum):
    UNCONFIGURED = 0
    IDLE = 1
    RUNNING = 2
    PAUSED = 3
    ERROR = -1


@dataclass
class StartStopHistogram:
    histogram: np.ndarray[tuple[int], np.dtype[np.int32]]
    total_count: int
    bin_width_ns: float
    exposure_time_s: float
    zero: int


class StartStopInterface(Base):
    @property
    @abstractmethod
    def inputs(self) -> Iterable[str]:
        """Return an iterable of input names."""

    @abstractmethod
    def configure_input(self, name: str, config: InputConfiguration) -> None:
        """Configure input `name` with `config`."""

    def configure_inputs(self, configs: dict[str, InputConfiguration]) -> None:
        """Configure all inputs"""
        for (name, config) in configs.items():
            self.configure_input(name, config)

    @abstractmethod
    def input_configuration(self, name: str) -> InputConfiguration:
        """Return the configuration of input `name`."""

    def input_configurations(self) -> dict[str, InputConfiguration]:
        """Return all input configurations."""
        return {name: self.input_configuration(name) for name in self.inputs}

    @property
    @abstractmethod
    def histograms(self) -> Iterable[str]:
        """Return an iterable of histogram names."""

    def is_histogram_enabled(self, histogram: str) -> bool:
        """Return `True` if an histogram will be acquired for the requested `histogram`."""
        config = self.histogram_configuration(histogram)
        return config.enabled

    def enable_histogram(self, histogram: str, enabled: bool) -> None:
        """Set the enable state of the given `histogram`."""
        config = self.histogram_configuration(histogram)
        config.enabled = enabled
        self.configure_histogram(histogram, config)

    def enable_histograms(self, states: dict[str, bool]) -> None:
        """Set the enable state of histograms specified by `states`."""
        for histogram, state in states.items():
            self.enable_histogram(histogram, state)

    @abstractmethod
    def histogram_configuration(self, histogram: str) -> StartStopHistogramConfiguration:
        """Return the current configuration of the given `histogram`."""

    @property
    def configuration(self) -> dict[str, StartStopHistogramConfiguration]:
        return {histogram: self.histogram_configuration(histogram) for histogram in self.histograms}

    @property
    def enabled_configuration(self) -> dict[str, StartStopHistogramConfiguration]:
        """Return the configuration of enabled histograms."""
        return {
            histogram: self.histogram_configuration(histogram)
            for histogram in self.histograms
            if self.is_histogram_enabled(histogram)
        }

    @abstractmethod
    def configure_histogram(self, histogram: str, config: StartStopHistogramConfiguration) -> None:
        """Configure the given `histogram`."""

    def configure(self, histograms: dict[str, StartStopHistogramConfiguration]) -> None:
        """Configure all histograms."""
        for histogram, configuration in histograms.items():
            self.configure_histogram(histogram, configuration)

    @property
    @abstractmethod
    def status(self) -> StartStopHardwareState:
        """Return the current status of the hardware."""

    @abstractmethod
    def start_acquisition(self) -> StartStopHardwareState:
        """Start acquisition on all enabled histogram histograms."""

    @abstractmethod
    def stop_acquisition(self) -> StartStopHardwareState:
        """Stop acquisition on all enabled histogram histograms."""

    @abstractmethod
    def pause_acquisition(self) -> StartStopHardwareState:
        """Pause acquisition on all enabled histogram histograms."""

    @abstractmethod
    def continue_acquisition(self) -> StartStopHardwareState:
        """Continue acquisition on all enabled histogram histograms."""

    @abstractmethod
    def histogram_data(self, histogram: str) -> StartStopHistogram:
        """Get the current data for histogram `histogram`."""

    @property
    def data(self) -> dict[str, StartStopHistogram]:
        """Get data on all histograms."""
        return {histogram: self.histogram_data(histogram) for histogram in self.histograms if self.is_histogram_enabled(histogram)}
