from enum import IntEnum
from abc import abstractmethod
from types import CodeType
from qudi.core.module import Base

class ScanningStateCode(IntEnum):
    SCANNING = 0
    IDLE = 1
    UNKNOWN = 2

class ScanningState:
    def __init__(self, code, d):
        self.code = code
        self.dict = d
    @classmethod
    def SCANNING(cls, d=None):
        ScanningState(ScanningStateCode.SCANNING, d)
    @classmethod
    def IDLE(cls, d=None):
        ScanningState(ScanningStateCode.IDLE, d)
    @classmethod
    def UNKNOWN(cls, d=None):
        ScanningState(ScanningStateCode.UNKNOWN, d)
    def is_scanning(self):
        return self.code == ScanningStateCode.SCANNING
    def is_idle(self):
        return self.code == ScanningStateCode.IDLE
    def is_unknown(self):
        return self.code == ScanningStateCode.UNKNOWN

class ScanningLaserReturnError(Exception):
    pass

class ScanningLaserInterface(Base):
    """This interface can be used to control a laser that can scan frequencies."""

    @abstractmethod
    def get_frequency(self):
        """Get current frequency of the laser.

        @return float: laser current frequency in THz
        """
        return 0.0

    @abstractmethod
    def set_frequency(self, value):
        """Set current frequency of the laser.

        @param float value: desired laser frequency
        """
        pass

    @abstractmethod
    def start_scan(self):
        """Start a scan on the laser.
        """
        pass

    @abstractmethod
    def set_scanning_points(self, frequencies):
        """Set the scanning points.

        @param list frequencies: list of frequencies which are to be scanned.
        """
        pass

    @abstractmethod
    def get_status(self):
        """Return the status of the laser.

        @return ScanningState state
        """
