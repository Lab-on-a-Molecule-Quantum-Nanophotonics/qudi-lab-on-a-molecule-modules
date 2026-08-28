 # -*- coding: utf-8 -*-

"""
Combine two hardware switches into one.

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

import time

from PySide6 import QtCore

from qudi.util.mutex import Mutex
from qudi.interface.switch_interface import SwitchInterface
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector

class SwitchPulseInterfuse(SwitchInterface):
    """ Methods to control slow (mechanical) laser switching devices.
    This interfuse in particular combines two switches into one.

    Example config for copy-paste:

    switch_pulser:
        module.Class: 'interfuse.switch_pulse_interfuse.SwitchPulseInterfuse'
        connect:
            switch: switch
        options:
            name: pulsed_switch  # optional name of the combined hardware
            # if True the switch names will be extended by the hardware name of the individual switches in front.
            pulse_length: 0.5

    """
    switch = Connector(interface=SwitchInterface)
    # optional name of the combined hardware
    _hardware_name = ConfigOption(name='name', default=None, missing='nothing')
    _pulse_length = ConfigOption(name='pulse_length', default=0.5, missing='nothing')
    _watchdog_delay = ConfigOption(name='watchdog_delay', default=0.1)
    
    sig_start_watchdog = QtCore.Signal()
    sig_stop_watchdog = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_start = {}
        self._watchdog_timer = QtCore.QTimer(parent=self)
        self._watchdog_lock = Mutex()
        self._watchdog_running = False
    
    def on_activate(self):
        """ Activate the module and fill status variables.
        """
        if self._hardware_name is None:
            self._hardware_name = self.module_name
        
        self.enable_watchdog()
        self.start_watchdog()

    def on_deactivate(self):
        """ Deactivate the module and clean up.
        """
        self.stop_watchdog()
        self.disable_watchdog()

    @property
    def name(self):
        """ Name of the hardware as string.

        @return str: The name of the hardware
        """
        return self._hardware_name

    @property
    def available_states(self):
        """ Names of the states as a dict of tuples.

        The keys contain the names for each of the switches. The values are tuples of strings
        representing the ordered names of available states for each switch.

        @return dict: Available states per switch in the form {"switch": ("state1", "state2")}
        """
        return self.switch().available_states

    @property
    def number_of_switches(self):
        """ Number of switches provided by the hardware.

        @return int: number of switches
        """
        return self.switch().number_of_switches

    @property
    def switch_names(self):
        """ Names of all available switches as tuple.

        @return str[]: Tuple of strings of available switch names.
        """
        return tuple(self.available_states)

    @property
    def states(self):
        """ The current states the hardware is in as state dictionary with switch names as keys and
        state names as values.

        @return dict: All the current states of the switches in the form {"switch": "state"}
        """
        return self.switch().states

    @states.setter
    def states(self, state_dict):
        """ The setter for the states of the hardware.

        The states of the system can be set by specifying a dict that has the switch names as keys
        and the names of the states as values.

        @param dict state_dict: state dict of the form {"switch": "state"}
        """
        assert isinstance(state_dict,
                          dict), f'Property "state" must be dict type. Received: {type(state_dict)}'
        current_states = self.switch1().states
        new_states = dict()
        for switch, state in state_dict.items():
            if switch in self._time_start: # the switch is already on
                new_states[switch] = current_states[switch]
            elif state != current_states[switch]:
                new_states[switch] = state
                with self._watchdog_lock:
                    self._time_start[switch] = (time.perf_counter(), current_states[switch])
            else:
                new_states[switch] = state
        self.switch().states = new_states

    def get_state(self, switch):
        """ Query state of single switch by name

        @param str switch: name of the switch to query the state for
        @return str: The current switch state
        """
        return self.switch().get_state(switch)

    def set_state(self, switch, state):
        """ Query state of single switch by name

        @param str switch: name of the switch to change
        @param str state: name of the state to set
        """
        current_state = self.get_state(switch)
        if state != current_state:
            with self._watchdog_lock:
                self._time_start[switch] = (time.perf_counter(), current_state)
        return self.switch().set_state(switch, state)
    
    def __start_watchdog(self) -> None:
        self._watchdog_running = True
        self._watchdog_timer.start(self._watchdog_delay * 1000)

    def __stop_watchdog(self) -> None:
        self._watchdog_running = False

    def start_watchdog(self) -> None:
        "Start the watchdog. The mechanism must have been enabled beforehand."
        self.sig_start_watchdog.emit()

    def stop_watchdog(self) -> None:
        "Interrupt the watchdog."
        self.sig_stop_watchdog.emit()
        
    def enable_watchdog(self) -> None:
        "Enable the watchdog mechanism."
        self._watchdog_timer.setSingleShot(True)
        self._watchdog_timer.timeout.connect(self.__watchdog, QtCore.Qt.QueuedConnection)
        self.sig_start_watchdog.connect(self.__start_watchdog, QtCore.Qt.QueuedConnection)
        self.sig_stop_watchdog.connect(self.__stop_watchdog, QtCore.Qt.QueuedConnection)

    def disable_watchdog(self) -> None:
        "Disable the watchdog mechanism. Will interrupt the watchdog if necessary."
        if self._watchdog_running:
            self.stop_watchdog()
        self._watchdog_timer.timeout.disconnect()
        self.sig_start_watchdog.disconnect(self.__start_watchdog)
        self.sig_stop_watchdog.disconnect(self.__stop_watchdog)
            
    def __watchdog(self):
        now = time.perf_counter()
        cleanup = set()
        for switch, (time_start, fallback_state) in self._time_start.items():
            if now - time_start >= self._pulse_length:
                self.switch().set_state(switch, fallback_state)
                cleanup.add(switch)
        with self._watchdog_lock:
            for switch in cleanup:
                self._time_start.pop(switch)
        if self._watchdog_running:
            self._watchdog_timer.start(self._watchdog_delay*1000)