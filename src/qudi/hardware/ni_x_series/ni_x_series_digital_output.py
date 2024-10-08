 # -*- coding: utf-8 -*-

"""
This file contains the qudi hardware module to use a National Instruments X-series card as mixed
signal input data streamer.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

# ToDo: Handle case where zero volts is not a good default value

import nidaqmx as ni
from PySide2 import QtCore

from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.helpers import natural_sort, in_range

from qudi.interface.switch_interface import SwitchInterface
from qudi.hardware.ni_x_series.helpers import sanitize_device_name, normalize_channel_name
from qudi.hardware.ni_x_series.helpers import do_line_names


class NIXSeriesDigitalOutput(SwitchInterface):
    _device_name = ConfigOption(name='device_name',
                                default='Dev1',
                                missing='warn',
                                constructor=sanitize_device_name)
    _default_pulse_length = ConfigOption(name='pulse_length',
                                default=2000)
    _channels_config = ConfigOption(
        name='channels',
        default = {},
        missing='warn'
    )

    _setpoints = StatusVar(name='current_setpoints', default=dict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = Mutex()
        self._device_channel_mapping = dict()
        self._do_task_handles = dict()
        self._keep_values = dict()
        self._on_values = dict()
        self._off_values = dict()
        self._channel_task_names = dict()
        self._pulse_length = dict()
        

    def on_activate(self):
        """ Starts up the NI-card and performs sanity checks """
        # Check if device is connected and set device to use
        self._device_channel_mapping = dict()
        self._do_task_handles = dict()
        self._keep_values = dict()
        self._on_values = dict()
        self._off_values = dict()
        self._channel_task_names = dict()
        self._pulse_length = dict()

        # Sanitize channel configuration
        valid_channels = do_line_names(self._device_name)
        valid_channels_lower = [name.lower() for name in valid_channels]
        for ch_name in natural_sort(self._channels_config):
            ch_cfg = self._channels_config[ch_name]
            channel = ch_cfg.get('line', ch_name)
            norm_name = normalize_channel_name(channel).lower()
            try:
                device_name = valid_channels[valid_channels_lower.index(norm_name)]
            except (ValueError, IndexError):
                self.log.error(f'Invalid analog output channel "{ch_name}" configured. Channel '
                               f'will be ignored.\nValid analog output channels are: '
                               f'{valid_channels}')
                continue
            self._device_channel_mapping[ch_name] = device_name
            self._channel_task_names[ch_name] = device_name.replace("/", "_")
            self._keep_values[ch_name] = bool(ch_cfg.get('keep_value', True))
            self._on_values[ch_name] = ch_cfg.get('on', "ON")
            self._off_values[ch_name] = ch_cfg.get('off', "OFF")
            self._pulse_length[ch_name] = ch_cfg.get('pulse_length', self._default_pulse_length if bool(ch_cfg.get('pulsed', False)) else None)
            self._init_do_task(ch_name)

        # Sanitize status variables
        self._sanitize_setpoint_status()
        for channel,value in self._setpoints.items():
            self.set_state(channel, value)

    def on_deactivate(self):
        for channel in list(self._do_task_handles):
            try:
                self._terminate_do_task(channel)
            except:
                self.log.exception(f'Error while terminating NI digital out task "{channel}":')

    # SwitchInterface 
    @property
    def name(self):
        return "NI switch: " + self._device_name
    
    @property
    def available_states(self):
        return {k:(self._off_values[k], self._on_values[k]) for k in self._channels_config.keys()}
        
    def get_state(self, switch):
        return self._setpoints[switch]
        
    def set_state(self, switch: str, state: str) -> None:
        self._setpoints[switch] = state
        if state == self._on_values[switch]:
            self.log.debug(f"Switching ON {switch}.")
            self._write_do_task(switch, True)
            if self._pulse_length[switch] is not None:
                QtCore.QTimer.singleShot(self._pulse_length[switch], lambda : self.set_state(switch, self._off_values[switch]))
        else:
            self.log.debug(f"Switching OFF {switch}.")
            self._write_do_task(switch, False)
        
    # Internal
    
    def _init_do_task(self, channel):
        if channel in self._do_task_handles:
            raise ValueError(f'AO task with name "{channel}" already present.')
        try:
            do_task = ni.Task(self._channel_task_names[channel])
        except ni.DaqError as err:
            raise RuntimeError(f'Unable to create NI task "{channel}"') from err
        # TODO: set that for ao
        try:
            do_phys_ch = f'/{self._device_name}/{self._device_channel_mapping[channel]}'
            do_task.do_channels.add_do_chan(do_phys_ch)
        except Exception as err:
            try:
                do_task.close()
            except ni.DaqError:
                pass
            raise RuntimeError('Error while configuring NI digital out task') from err
        self._do_task_handles[channel] = do_task
        
    def _write_do_task(self, channel: str, value: bool):
        self._do_task_handles[channel].write(value)
        
    def _terminate_do_task(self, channel: str) -> None:
        try:
            if not self._keep_values[channel]:
                self._write_do_task(channel, False)
            task = self._do_task_handles.pop(channel)
        except KeyError:
            return
        try:
            if not task.is_task_done():
                task.stop()
        finally:
            task.close()
            
    def _sanitize_setpoint_status(self) -> None:
        # Remove obsolete channels and out-of-bounds values
        available_states = self.available_states
        for channel, value in list(self._setpoints.items()):
            try:
                if not value in available_states[channel]:
                    del self._setpoints[channel]
            except KeyError:
                del self._setpoints[channel]
        # Add missing setpoint channels and set initial value to zero
        self._setpoints.update(
            {ch: available[0] for (ch, available) in available_states.items() if ch not in self._setpoints}
        )
