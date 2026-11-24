
import elliptec
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import Mutex

from qudi.interface.process_control_interface import ProcessControlConstraints, ProcessSetpointInterface


class ElliptecRotator(ProcessSetpointInterface):

    comport = ConfigOption(
        name='comport',
        default='COM5'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._rotator = None
        self._activity_states = dict()
        self._constraints = None

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        units = {'angle':'deg'}
        limits = {'angle': (0,360)}
        dtypes = {'angle': float}
        self._constraints = ProcessControlConstraints(
            setpoint_channels=['angle'],
            process_channels=[],
            units=units,
            limits=limits,
            dtypes=dtypes
        )
        controller = elliptec.Controller(self.comport)
        self._rotator = elliptec.Rotator(controller)
        self._activity_states = {ch: False for ch in self._constraints.all_channels}

    def on_deactivate(self):
        self._activity_states = {ch: False for ch in self.constraints.all_channels}
        if self._rotator is not None:
            self._rotator = None

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
        try:
            active = bool(active)
        except Exception as err:
            raise TypeError('Unable to convert activity state to bool') from err
        with self._thread_lock:
            try:
                current_state = self._get_activity_state(channel)
            except KeyError as err:
                raise ValueError(f'Invalid channel specifier "{channel}". Valid channels are:\n'
                                 f'{self.constraints.all_channels}') from err
            if active != current_state:
                self._activity_states[channel] = active
                self._update_module_state()

    def _update_module_state(self) -> None:
        module_busy = any(state for state in self._activity_states.values())
        if module_busy and self.module_state() != 'locked':
            self.module_state.lock()
        elif not module_busy and self.module_state() == 'locked':
            self.module_state.unlock()

    def get_activity_state(self, channel: str) -> bool:
        """ Get activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        with self._thread_lock:
            return self._get_activity_state(channel)

    def _get_activity_state(self, channel: str) -> bool:
        try:
            return self._activity_states[channel]
        except KeyError as err:
            raise ValueError(f'Invalid channel specifier "{channel}". Valid channels are:\n'
                             f'{self.constraints.all_channels}') from err

    def set_setpoint(self, channel: str, value: float) -> None:
        """ Set new setpoint for a single channel """
        with self._thread_lock:
            try:
                if not self.constraints.channel_value_in_range(channel, value)[0]:
                    raise ValueError(f'Setpoint {value} for channel "{channel}" out of allowed '
                                     f'value bounds {self.constraints.channel_limits[channel]}')
                if channel == "angle" and self._rotator is not None:
                    self._rotator.set_angle(value)
            except KeyError as err:
                raise ValueError(f'Invalid setpoint channel specifier "{channel}". Valid setpoint '
                                 f'channels are:\n{tuple(self._setpoints)}') from err

    def get_setpoint(self, channel: str) -> int | float:
        """ Get current setpoint for a single channel """
        with self._thread_lock:
            try:
                if self._rotator is not None:
                    return self._rotator.get_angle()
                raise ValueError("channel not active")
            except KeyError as err:
                raise ValueError(f'Invalid setpoint channel specifier "{channel}". Valid setpoint '
                                 f'channels are:\n{tuple(self._setpoints)}') from err


