"""
Qudi interface definitions for a simple multi/single channel setpoint device,
a simple multi/single channel process value reading device
and the combination of both (reading/setting setpoints and reading process value).

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ['ProcessSetpointInterface', 'ProcessValueInterface', 'ProcessControlInterface',
           'ProcessControlConstraints']

from abc import abstractmethod
from typing import Iterable, Mapping, Union, Optional, Tuple, Type, Dict
import numpy as np

from qudi.core.module import Base

class SimpleScannerConstraints:
    def __init__(self, scan_range, sample_rate_range, number_of_samples_range):
        self.scan_min = min(scan_range)
        self.scan_max = max(scan_range)
        self.sample_rate_min = min(sample_rate_range)
        self.sample_rate_max = max(sample_rate_range)
        self.number_of_samples_min = min(number_of_samples_range)
        self.number_of_samples_max = max(number_of_samples_range)
        
class SimpleScannerSettings:
    def __init__(self, first_point, last_point, sample_rate, number_of_samples):
        self.first_point = first_point
        self.last_point = last_point
        self.sample_rate = sample_rate
        self.number_of_samples = number_of_samples
    def is_compatible_with(self, constraints: SimpleScannerConstraints) -> Tuple[bool, Iterable[str]]:
        faulty_settings = []
        faulty = False
        if not (constraints.scan_min <= self.first_point <= constraints.scan_max):
            faulty_settings.append("first_point")
        if not (constraints.scan_min <= self.last_point <= constraints.scan_max):
            faulty_settings.append("last_point")
        if not (constraints.sample_rate_min <= self.sample_rate <= constraints.sample_rate_max):
            faulty_settings.append("sample_rate")
        if not (constraints.number_of_samples_min <= self.number_of_samples <= constraints.number_of_samples_max):
            faulty_settings.append("number_of_samples")
        return (len(faulty_settings) == 0), faulty_settings

class SimpleScannerInterface(Base):
    """
    """
    @property
    @abstractmethod
    def constraints(self) -> SimpleScannerConstraints:
        """ Read-Only property holding the constraints for this hardware module.
        See class ProcessControlConstraints for more details.
        """
        pass
    @property
    @abstractmethod 
    def settings(self) -> SimpleScannerSettings:
        pass
    @abstractmethod 
    def set_settings(self, settings: SimpleScannerSettings) -> Tuple[bool, Iterable[str]]:
        pass
    @abstractmethod
    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
    @abstractmethod
    def get_setpoint(self) -> float:
        pass
    @abstractmethod
    def set_setpoint(self, value: float) -> None:
        pass
    @abstractmethod
    def reset_continuous(self):
        """Reset the scanner to normal operation mode, i.e. not scanning (e.g. continuous reading of wavelength)."""