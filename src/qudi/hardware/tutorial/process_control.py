import time
import numpy as np
from typing import Union

from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.interface.process_control_interface import ProcessControlConstraints
from qudi.interface.process_control_interface import ProcessControlInterface
from qudi.interface.mixins.process_control_switch import ProcessControlSwitchMixin

from .physical_model import PhysicalModel


class ProcessControlTutorial(ProcessControlInterface):
    """

    Example config for copy-paste:

    ```yaml
    process_control:
        module.Class: 'tutorial.process_control.ProcessControlTutorial'
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._constraints = None
        self.model = PhysicalModel()

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._constraints = ProcessControlConstraints(
            setpoint_channels=['laser frequency'],
            process_channels=['APD'],
            units={'laser frequency': 'Hz', 'APD':'Hz'},
            limits={'laser frequency':(-200e9, 200e9)},
            dtypes={'laser frequency':float, 'APD':float}
        )

    def on_deactivate(self):
        pass

    @property
    def constraints(self) -> ProcessControlConstraints:
        """ Read-Only property holding the constraints for this hardware module.
        See class ProcessControlConstraints for more details.
        """
        return self._constraints

    def set_activity_state(self, channel: str, active: bool) -> None:
        """ Set activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        pass

    def get_activity_state(self, channel: str) -> bool:
        """ Get activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        return True

    def get_process_value(self, channel: str) -> Union[int, float]:
        """ Get current process value for a single channel """
        if channel == "APD":
            return self.model()
        else:
            raise ValueError(f"Unknown process {channel}")

    def set_setpoint(self, channel: str, value: Union[int, float]) -> None:
        """ Set new setpoint for a single channel """
        if channel == "laser frequency":
            self.model.idle = value
        else:
            raise ValueError(f"Unknown setpoint {channel}")

    def get_setpoint(self, channel: str) -> Union[int, float]:
        """ Get current setpoint for a single channel """
        if channel == "laser frequency":
            return self.model.idle
        else:
            raise ValueError(f"Unknown setpoint {channel}")

