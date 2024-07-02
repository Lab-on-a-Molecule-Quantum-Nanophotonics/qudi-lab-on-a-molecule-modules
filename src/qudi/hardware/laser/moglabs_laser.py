# -*- coding: utf-8 -*-
"""
This module controls the MOGLabs laser.

Copyright (c) 2024, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution.

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

import socket
import queue
import struct
import time
from typing import Union, Optional, Tuple, Sequence

from PySide2 import QtCore
from PySide2.QtGui import QGuiApplication

import serial
import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
# from qudi.interface.scanning_laser_interface import ScanningLaserInterface, ScanningState, ScanningLaserReturnError
from qudi.interface.data_instream_interface import DataInStreamInterface, DataInStreamConstraints, SampleTiming, StreamingMode, ScalarConstraint
from qudi.interface.process_control_interface import ProcessControlConstraints, ProcessControlInterface
from qudi.interface.autoscan_interface import AutoScanInterface, AutoScanConstraints
from qudi.interface.switch_interface import SwitchInterface
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from qudi.interface.scanning_probe_interface import ScanningProbeInterface, ScanData, ScannerChannel, ScannerAxis, ScanConstraints
from qudi.util.mutex import Mutex
from qudi.core.connector import Connector
from qudi.util.overload import OverloadedAttribute
from qudi.util.helpers import in_range
from qudi.util.enums import SamplingOutputMode

from qudi.hardware.laser.moglabs_helper import MOGLABSDeviceFinder

class MOGLABSMotorizedLaserDriver(SwitchInterface, ProcessControlInterface):
    """
    Control the Laser diode driver directly.
    """
    port = ConfigOption("port")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serial = serial.Serial()

    # Qudi activation / deactivation
    def on_activate(self):
        """Activate module.
        """
        device_finder = MOGLABSDeviceFinder()
        self.serial = device_finder.ldd
        self._ramp_running = False
        self._ramp_halt = 0.0
        self._lock = Mutex()     
        self.serial.open()
        self._set_ramp_status("OFF")

    def on_deactivate(self):
        """Deactivate module.
        """
        self.serial.close()

    # SwitchInterface
    @property
    def name(self):
        return "LDD"
    @property
    def available_states(self):
        return {
                "HV,MOD":("EXT", "RAMP"),
                "Temp. control":("OFF", "ON"),
                "Curr. control":("OFF", "ON"),
                "RAMP":("OFF", "ON"),
                "CURRENT,MOD":("OFF", "+RAMP"),
        }
    def get_state(self, switch):
        with self._lock:
            if switch == "HV,MOD":
                return self._mod_status()
            elif switch == "Temp. control":
                return self._temp_status()
            elif switch == "RAMP":
                return self._ramp_status()
            elif switch == "CURRENT,MOD":
                return self._get_current_mod()
            else:
                return self._current_status()

    def set_state(self, switch, state):
        with self._lock:
            if switch == "HV,MOD":
                self._set_mod_status(state)
            elif switch == "Temp. control":
                self._set_temp_status(state)
            elif switch == "RAMP":
                self._set_ramp_status(state)
            elif switch == "CURRENT,MOD":
                return self._set_current_mod(state)
            else:
                self._set_current_status(state)
            
    # ProcessControlInterface
    def set_setpoint(self, channel, value):
        with self._lock:
            if channel == "frequency":
                return self._set_freq(value)
            elif channel == "span":
                return self._set_span(value)
            elif channel == "offset":
                return self._set_offset(value)
            elif channel == "bias":
                return self._set_bias(value)
            elif channel == "duty":
                return self._set_duty(value)
            elif channel == "ramp_halt": 
                self._ramp_halt=value

    def get_setpoint(self, channel):
        with self._lock:
            if channel == "frequency":
                return self._get_freq()
            elif channel == "span":
                return self._get_span()
            elif channel == "offset":
                return self._get_offset()
            elif channel == "bias":
                return self._get_bias()
            elif channel == "duty":
                return self._get_duty()
            elif channel == "ramp_halt": 
                return self._ramp_halt

    def get_process_value(self, channel):
        with self._lock:
            return self._get_current()

    def set_activity_state(self, channel, active):
        """ Set activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        pass

    def get_activity_state(self, channel):
        """ Get activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        return True

    @property
    def constraints(self):
        """ Read-Only property holding the constraints for this hardware module.
        See class ProcessControlConstraints for more details.
        """
        with self._lock:
            return ProcessControlConstraints(
                ["frequency", "span", "offset", "bias", "duty", "ramp_halt"],
                ["current"],
                {
                    "frequency":"Hz",
                    "span":"",
                    "offset":"",
                    "bias":"mA",
                    "duty":"",
                    "current":"mA",
                    "ramp_halt":"",
                },
                {
                    "frequency":(0.0, 50),
                    "span":(0.0,1.0),
                    "offset":(0.0,1.0),
                    "bias":(0.0,50.0),
                    "duty":(0.0,1.0),
                    "current":(0.0,self._get_current_lim()),
                    "ramp_halt":(0.0,1.0),
                },
                {
                    "frequency":float,
                    "span":float,
                    "offset":float,
                    "bias":float,
                    "duty":float,
                    "current":float,
                    "ramp_halt":float,
                },
            )
        
    # Internal communication facilities
    def send_and_recv(self, value, check_ok=True):
        if not value.endswith("\r\n"):
            value += "\r\n"
        self.serial.write(value.encode("utf8"))
        ret = self.serial.readline().decode('utf8')
        if check_ok and not ret.startswith("OK"):
            self.log.error(f"Command \"{value}\" errored: \"{ret}\"")
        return ret

    def _mod_status(self):
        return self.send_and_recv("hv,mod", check_ok=False).rstrip()
        
    def _set_mod_status(self, val):
        return self.send_and_recv(f"hv,mod,{val}")

    def _temp_status(self):
        return self.send_and_recv("TEC,ONOFF", check_ok=False).rstrip()
        
    def _set_temp_status(self, val):
        return self.send_and_recv(f"TEC,ONOFF,{val}")

    def _current_status(self):
        return self.send_and_recv("CURRENT,ONOFF", check_ok=False).rstrip()
        
    def _set_current_status(self, val):
        return self.send_and_recv(f"CURRENT,ONOFF,{val}")

    def _set_freq(self, val):
        return self.send_and_recv(f"RAMP,FREQ,{val}")
    def _get_freq(self):
        return float(self.send_and_recv(f"RAMP,FREQ", check_ok=False).split()[0])
    def _set_span(self, val):
        return self.send_and_recv(f"RAMP,SPAN,{val}")
    def _get_span(self):
        return float(self.send_and_recv(f"RAMP,SPAN", check_ok=False).split()[0])
    def _set_offset(self, val):
        return self.send_and_recv(f"RAMP,OFFSET,{val}")
    def _get_offset(self):
        return float(self.send_and_recv(f"RAMP,OFFSET", check_ok=False).split()[0])
    def _set_bias(self, val):
        return self.send_and_recv(f"RAMP,BIAS,{val}")
    def _get_bias(self):
        return float(self.send_and_recv(f"RAMP,BIAS", check_ok=False).split()[0])
    def _set_duty(self, val):
        return self.send_and_recv(f"RAMP,DUTY,{val}")
    def _get_duty(self):
        return float(self.send_and_recv(f"RAMP,DUTY", check_ok=False).split()[0])
    def _get_current_lim(self):
        return float(self.send_and_recv(f"current,ilim", check_ok=False).split()[0])
    def _get_current(self):
        return float(self.send_and_recv(f"current,meas", check_ok=False).split()[0])
    def _set_ramp_status(self, st):
        if st == "OFF":
            self._ramp_running = False
            self.send_and_recv(f"ramp,halt,{self._ramp_halt}", check_ok=False)
        else:
            self._ramp_running = True
            self.send_and_recv(f"ramp,resume")
    def _ramp_status(self):
        if self._ramp_running:
            return "ON"
        else:
            return "OFF"
    def _get_current_mod(self):
        return self.send_and_recv("CURRENT,MOD", check_ok=False).rstrip()
    def _set_current_mod(self, value):
        return self.send_and_recv(f"CURRENT,MOD,{value}")

class MOGLABSCateyeLaser(ProcessControlInterface, AutoScanInterface):
    _scan_duration = StatusVar(name="scan_duration", default=1.0)
    __sigResetMotor = QtCore.Signal()
    _last_scan_pd = StatusVar(name="last_scan_pd", default=np.zeros(0, dtype=float))
    _last_scan_piezo = StatusVar(name="last_scan_piezo", default=np.zeros(0, dtype=int))
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serial = serial.Serial()

    # Qudi activation / deactivation
    def on_activate(self):
        """Activate module.
        """
        device_finder = MOGLABSDeviceFinder()
        self.serial = device_finder.cem
        self.serial.open()
        self._lock = Mutex()     
        self.__sigResetMotor.connect(self.__reset_motor, QtCore.Qt.QueuedConnection)
        self._reset_motor()

    def on_deactivate(self):
        """Deactivate module.
        """
        self.serial.close()
    def set_setpoint(self, channel, value):
        with self._lock:
            if channel=="grating":
                self._set_motor_position(value)
            else:
                self._scan_duration=value

    def get_setpoint(self, channel):
        with self._lock:
            if channel=="grating":
                return self._get_motor_setpoint()
            else:
                return self._scan_duration

    def get_process_value(self, channel):
        with self._lock:
            if channel=="grating":
                return self._motor_position()
            elif channel=="photodiode":
                return self._get_pd()
            else:
                return self._get_piezo()

    def set_activity_state(self, channel, active):
        """ Set activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        pass

    def get_activity_state(self, channel):
        """ Get activity state for given channel.
        State is bool type and refers to active (True) and inactive (False).
        """
        return True

    constraints = OverloadedAttribute()
    @constraints.overload("ProcessControlInterface")
    @property
    def constraints(self):
        """ Read-Only property holding the constraints for this hardware module.
        See class ProcessControlConstraints for more details.
        """
        return ProcessControlConstraints(
            ["grating", "scan_duration"],
            ["grating", "photodiode", "piezo"],
            {"grating":"step", "photodiode":"V", "piezo":"V", "scan_duration":"s"},
            {"grating":self._motor_range(), "scan_duration":(0.1,20)},
            {"grating":int, "photodiode":float, "piezo":float, "scan_duration":float},
        )

    # AutoScanInterface
    @constraints.overload("AutoScanInterface")
    @property
    def constraints(self):
        return AutoScanConstraints(
            channels=["photodiode", "piezo"],
            units={"photodiode":"V", "piezo":""},
            limits={"photodiode":(0,5), "piezo":(0,2**16)},
            dtypes={"photodiode":float, "piezo":int}
        )
    def trigger_scan(self):
        with self._lock:
            vals = self._scan_pd()
            self._last_scan_pd = vals[0,:]*5.0/(2**12-1)
            self._last_scan_piezo = vals[1,:]

    def get_last_scan(self, channel):
        if channel == "photodiode":
            return self._last_scan_pd
        else:
            return self._last_scan_piezo
        
    # Internal communication facilities
    def send_and_recv(self, value, check_ok=True):
        if not value.endswith("\r\n"):
            value += "\r\n"
        self.serial.write(value.encode("utf8"))
        ret = self.serial.readline().decode('utf8')
        if check_ok and not ret.startswith("OK"):
            self.log.error(f"Command \"{value}\" errored: \"{ret}\"")
        return ret

    def _motor_range(self):
        mini,maxi = self.send_and_recv("motor,travel", check_ok=False).split(" ")
        return int(mini), int(maxi)
    
    def _motor_position(self):
        return int(self.send_and_recv("motor,position", check_ok=False))
        
    def _set_motor_position(self, value):
        return self.send_and_recv(f"motor,dest,{int(value)}")
        
    def _get_motor_setpoint(self):
        return int(self.send_and_recv(f"motor,dest", check_ok=False))
        
    def _move_motor_rel(self, value):
        return self.send_and_recv(f"motor,step,{int(value)}")

    def _motor_status(self):
        return self.send_and_recv("motor,status", check_ok=False).rstrip()
        
    def _reset_motor(self):
        self.__sigResetMotor.emit()
        
    def __reset_motor(self):
        with self._lock:
            old_setpoint = self._get_motor_setpoint()
            self.send_and_recv("motor,home")
            self.log.info("Please wait while the grating is being homed.")
            while self._motor_status() not in ("STABILISING", "ERR STATE"):
                time.sleep(0.01)
            if self._motor_status() == "ERR STATE":
                self.log.error("Motor is in error state! Consider restarting the CEM.")
            else:
                self.log.debug(f"Seting CEM setpoint to {old_setpoint}")
                self._set_motor_position(old_setpoint)
                while np.abs(self._motor_position() - old_setpoint) > 1:
                    time.sleep(0.01)
                self.log.info("Homing done.")
                
    def _get_pd(self):
        return float(self.send_and_recv("pd,read,0", check_ok=False).split()[0])

    def _get_piezo(self):
        return float(self.send_and_recv("pd,read,1", check_ok=False).split()[0])
        
    def _scan_pd(self,duration=None):
        if duration is None:
            duration = self._scan_duration
        self.send_and_recv("pd,rate,1")
        self.serial.read(self.serial.in_waiting)
        old_timeout = self.serial.timeout
        self.serial.timeout = duration * 1.5
        cmd = f"pd,scan,{duration}\r\n"
        self.serial.write(cmd.encode("utf8"))
        l = struct.unpack("<I", self.serial.read(4))[0]
        vals = np.empty(l//2, int)
        binary_data = self.serial.read(l)
        if len(binary_data) == 0:
            self.log.warning(f"I read 0")
            binary_data = self.serial.read(l)
        if len(binary_data) < l:
            self.log.warning(f"I did not read everything. read {len(binary_data)}/{l}.")
            binary_data += self.serial.read(l)
        self.log.debug(f"Read {len(binary_data)}/{l}")
        for i in range(0, l, 2):
            vals[i//2] = struct.unpack("<H", binary_data[i:i+2])[0]
        self.serial.timeout = old_timeout
        return np.reshape(vals, (2, len(vals)//2), 'F')

class MOGLabsFZWScanner(ScanningProbeInterface):
    """A class to control our MOGLabs laser to perform excitation spectroscopy.

    Example config:

    """
    cem = Connector(name="cateye_laser", interface="ProcessControlInterface")
    ldd = Connector(name="laser_driver", interface="SwitchInterface")
    fzw_process_control = Connector(name="wavemeter_process", interface="ProcessControlInterface")
    counter = Connector(name="counter", interface="FiniteSamplingInputInterface")

    _counter_apd_channel = ConfigOption(name="counter_apd_channel", missing="error")
    _apd_oversampling = ConfigOption(name="apd_oversampling", default=2)
    
    _threaded = True  # Interfuse is by default not threaded.

    sigNextDataChunk = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.axes=[]
        self.channels=[]
        self.constraints=None
        self._scan_data = None

        self._current_scan_frequency = -1
        self._current_scan_ranges = [tuple(), tuple()]
        self._current_scan_axes = tuple()
        self._current_scan_resolution = tuple()

        self._target_pos = dict()
        self._stored_target_pos = dict()

        self._min_step_interval = 1e-3
        self._scanner_distance_atol = 1
        self._start_scan_after_cursor = False
        self._abort_cursor_move = True
        self.__t_last_follow = None
        
        self._grating_positions = None
        self._tension_positions = None
        self._grating_position_index = 0
        self._tension_position_index = 0

        self.__write_timer = None
        self._thread_lock_cursor = Mutex()
        self._thread_lock_data = Mutex()

    def on_activate(self):
        assert self._apd_oversampling >= 2, "APD oversampling should be at least 2."
        constraints_process_control = self.fzw_process_control().constraints
        self.axes = [
            ScannerAxis(
                name="grating",
                unit="step",
                value_range=self.cem().constraints.channel_limits["grating"],
                step_range=(0, 200),
                resolution_range=(1,10000),
            ),
            ScannerAxis(
                name="tension",
                unit="V",
                value_range=constraints_process_control.channel_limits["tension"],
                step_range=(0, 2.5),
                resolution_range=(1, 1e9),
            ),
        ]
        self.channels = [
            ScannerChannel(
                name="APD",
                unit="c/s",
            ),
        ]
        for channel in constraints_process_control.process_channels:
            self.channels.append(
                ScannerChannel(
                    name=channel,
                    unit=constraints_process_control.channel_units[channel],
                )
              )
        self.constraints = ScanConstraints(
            axes=self.axes,
            channels=self.channels,
            backscan_configurable=False,
            has_position_feedback=True,
            square_px_only=False,
        )

        self._target_pos = self.get_position()  
        self._t_last_move = time.perf_counter()
        self.__init_write_timer()
        self.sigNextDataChunk.connect(self._fetch_data_chunk, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        self._abort_cursor_movement()
        if self.counter().module_state() == "locked":
            self.counter().stop_buffered_frame()

    # Internal facilities
    def _shrink_scan_ranges(self, ranges, factor=0.01):
        lenghts = [stop - start for (start, stop) in ranges]
        return [(start + factor* lenghts[idx], stop - factor* lenghts[idx]) for idx, (start, stop) in enumerate(ranges)]

    def _create_scan_data(self, axes, ranges, resolution, frequency):
        valid_scan_grid = False
        i_trial, n_max_trials = 0, 25

        while not valid_scan_grid and i_trial < n_max_trials:
            if i_trial > 0:
                ranges = self._shrink_scan_ranges(ranges)
            scan_data = ScanData(
                channels=tuple(self.constraints.channels.values()),
                scan_axes=tuple(self.constraints.axes[ax] for ax in axes),
                scan_range=ranges,
                scan_resolution=tuple(resolution),
                scan_frequency=frequency,
                position_feedback_axes=None)
            try:
                tension_scan_array,grating_scan_array = self._init_scan_arrays(scan_data)
                valid_scan_grid = True
            except ValueError:
                valid_scan_grid = False
            i_trial += 1
        if not valid_scan_grid:
            raise ValueError("Couldn't create scan grid. ")
        if i_trial > 1:
            self.log.warning(f"Adapted out-of-bounds scan range to {ranges}")
        # self.log.debug(f"New scanData created: {self._scan_data.data}")
        return scan_data

    def _init_scan_arrays(self, scan_data):
        """
        @param ScanData scan_data: The desired ScanData instance
        """
        assert isinstance(scan_data, ScanData), 'This function requires a scan_data object as input'

        scan_coords = dict()
        for index,axis in enumerate(scan_data.scan_axes):
            resolution = scan_data.scan_resolution[index]
            stop_points = np.linspace(scan_data.scan_range[index][0], scan_data.scan_range[index][1],
                                     resolution)
            scan_coords[axis] = stop_points
        self._check_scan_grid(scan_coords)
        tension_positions = scan_coords.get("tension", [self.get_position()["tension"]])
        grating_positions = scan_coords.get("grating", [self.get_position()["grating"]])

        return tension_positions, grating_positions

    def _check_scan_grid(self, scan_coords):
        for ax, coords in scan_coords.items():
            position_min = self.get_constraints().axes[ax].min_value
            position_max = self.get_constraints().axes[ax].max_value
            out_of_range = any(coords < position_min) or any(coords > position_max)
            if out_of_range:
                raise ValueError(f"Scan axis {ax} out of range [{position_min}, {position_max}]")

    @property
    def scan_settings(self):

        settings = {'axes': tuple(self._current_scan_axes),
                    'range': tuple(self._current_scan_ranges),
                    'resolution': tuple(self._current_scan_resolution),
                    'frequency': self._current_scan_frequency}
        return settings

    @property
    def is_scan_running(self):
        """
        Read-only flag indicating the module state.

        @return bool: scanning probe is running (True) or not (False)
        """
        # module state used to indicate hw timed scan running
        #self.log.debug(f"Module in state: {self.module_state()}")
        #assert self.module_state() in ('locked', 'idle')  # TODO what about other module states?
        if self.module_state() == 'locked':
            return True
        else:
            return False

    @property
    def is_move_running(self):
        with self._thread_lock_cursor:
            running = self.__t_last_follow is not None
            return running

    def __wait_on_move_done(self):
        try:
            t_start = time.perf_counter()
            while self.is_move_running:
                self.log.debug(f"Waiting for move done: {self.is_move_running}, {1e3*(time.perf_counter()-t_start)} ms")
                QGuiApplication.processEvents()
                time.sleep(self._min_step_interval)

            #self.log.debug(f"Move_abs finished after waiting {1e3*(time.perf_counter()-t_start)} ms ")
        except:
            self.log.exception("")

    def _prepare_movement(self, position, velocity=None):
        """
        Clips values of position to allowed range and fills up the write queue.
        If re-entered from a different thread, clears current write queue and start
        a new movement.
        """

        with self._thread_lock_cursor:
            self._abort_cursor_move = False
            constr = self.get_constraints()

            for axis, pos in position.items():
                in_range_flag, _ = in_range(pos, *constr.axes[axis].value_range)
                if not in_range_flag:
                    position[axis] = float(constr.axes[axis].clip_value(position[axis]))
                    self.log.warning(f'Position {pos} out of range {constr.axes[axis].value_range} '
                                     f'for axis {axis}. Value clipped to {position[axis]}')
                # TODO Adapt interface to use "in_range"?
                self._target_pos[axis] = position[axis]
            
                
            if self.fzw_process_control().get_setpoint("tension") != self._target_pos["tension"]:
                self.fzw_process_control().set_setpoint("tension", self._target_pos["tension"])
            if self.cem().get_setpoint("grating") != self._target_pos["grating"]:
                now = time.perf_counter()
                if now - self._t_last_move < 0.05: # Don't flood the motor with commands
                    time.sleep(0.05)
                self.cem().set_setpoint("grating", self._target_pos["grating"])
                self._t_last_move = time.perf_counter()
                
    def _reset_motor(self):
        self.cem()._reset_motor()

    def __init_write_timer(self):
        self.__write_timer = QtCore.QTimer(parent=self)

        self.__write_timer.setSingleShot(True)
        self.__write_timer.timeout.connect(self.__cursor_write_loop, QtCore.Qt.QueuedConnection)
        self.__write_timer.setInterval(1e3*self._min_step_interval)  # (ms), dynamically calculated during write loop

    def __start_write_timer(self):
        #self.log.debug(f"ao start write timer in thread {self.thread()}, QT.QThread {QtCore.QThread.currentThread()} ")
        try:
            if not self.is_move_running:
                #self.log.debug("Starting AO write timer...")
                if self.thread() is not QtCore.QThread.currentThread():
                    QtCore.QMetaObject.invokeMethod(self.__write_timer,
                                                    'start',
                                                    QtCore.Qt.BlockingQueuedConnection)
                else:
                    self.__write_timer.start()
            else:
                pass
                #self.log.debug("Dropping timer start, already running")

        except:
            self.log.exception("")
    
    def _setpoint_reached(self):    
        current_pos = self.get_position()
        with self._thread_lock_cursor:
            grating_setpoint = self._target_pos["grating"]
            grating_pos = current_pos["grating"]
            distance_to_target = np.abs(grating_setpoint - grating_pos)
            reached = distance_to_target <= self._scanner_distance_atol
            #if not reached:
            #    self.log.debug(f"Setpoint not reached. Delta: {distance_to_target}")
            return reached

    def __cursor_write_loop(self):
        t_start = time.perf_counter()
        try:
            stop_loop = self._setpoint_reached()
            with self._thread_lock_cursor:
                stop_loop = stop_loop or self._abort_cursor_move
                if not stop_loop:
                    if not self.__t_last_follow:
                        self.__t_last_follow = t_start
                    t_overhead = time.perf_counter() - t_start
                    self.__write_timer.start(int(round(1000 * max(0, self._min_step_interval - t_overhead))))
            if stop_loop:              
                self._abort_cursor_movement()
                if self._start_scan_after_cursor:
                    self._start_hw_timed_scan()
        except:
            self.log.exception("Error in ao write loop: ")

    def _pos_dict_to_vec(self, position):
        pos_list = [el[1] for el in sorted(position.items())]
        return np.asarray(pos_list)

    def _abort_cursor_movement(self):
        """
        Abort the movement.
        """

        #self.log.debug(f"Aborting move.")
        self._target_pos = self.get_position()

        with self._thread_lock_cursor:
            self._abort_cursor_move = True
            self.__t_last_follow = None

    def _move_to_and_start_scan(self, position):
        self._prepare_movement(position)
        self._start_scan_after_cursor = True
        #self.log.debug("Starting timer to move to scan position")
        self.__start_write_timer()

    def _start_hw_timed_scan(self):
        try:
            self.log.debug("start hw scan.")
            self.sigNextDataChunk.emit()
        except Exception as e:
            self.log.error(f'Could not start frame due to {str(e)}')
            self.module_state.unlock()

        self._start_scan_after_cursor = False

    def _current_data_index(self):
        scan_axes = self._scan_data.scan_axes
        if scan_axes == ("grating", "tension"):
            return (self._grating_position_index, self._tension_position_index)
        if scan_axes == ("tension", "grating"):
            return (self._tension_position_index, self._grating_position_index)
        elif "grating" in scan_axes:
            return self._grating_position_index
        elif "tension" in scan_axes:
            return self._tension_position_index
        else:
            raise ValueError(f"Inconsistent scan_axes for indexing: {scan_axes}")

    def _next_setpoint_indices(self):
        scan_axes = set(self._scan_data.scan_axes)
        if len(scan_axes) == 2:
            if self._tension_position_index >= len(self._tension_positions)-1:
                self.log.debug("finished a line!")
                self._tension_position_index = 0
                self._grating_position_index += 1
            else:
                self._tension_position_index += 1
        elif "tension" in scan_axes:
            self._tension_position_index += 1
        elif "grating" in scan_axes:
            self._grating_position_index += 1
        else:
            raise ValueError(f"Un-handled scan axes {scan_axes}")
            
    def _all_points_scanned(self):
        return self._tension_position_index >= len(self._tension_positions) or self._grating_position_index >= len(self._grating_positions)

    def _fetch_data_chunk(self):
        try:
            if not self._setpoint_reached():
                # We are taking a suspicious amount of time to move. Let's reset the motor.
                now = time.perf_counter()
                if np.abs(self.cem().get_setpoint("grating") - self.cem().get_process_value("grating")) > 1 and now - self._t_last_move > 10:
                    self.log.debug("The motor is stuck, I will reset it.")
                    self._reset_motor()
                    time.sleep(0.05)
                if not self.is_move_running and now - self._t_last_move > 2:
                    self.log.debug("setpoint not reached but no move is running.")
                    self._prepare_movement({
                        "grating" : self._grating_positions[self._grating_position_index],
                        "tension" : self._tension_positions[self._tension_position_index],
                    })
                    self.__start_write_timer()
                self.sigNextDataChunk.emit()
                return
            else:
                constraints_process_control = self.fzw_process_control().constraints
                process_channels = constraints_process_control.process_channels
                counts = self.counter().acquire_frame(frame_size=self._apd_oversampling)
                process_values = {ch: self.fzw_process_control().get_process_value(ch) for ch in process_channels}
                counts = counts[self._counter_apd_channel].mean()
                data_index = self._current_data_index()
                with self._thread_lock_data:
                    self._scan_data.data["APD"][data_index] = counts
                    for (k,v) in process_values.items():
                        self._scan_data.data[k][data_index] = v
                    self._next_setpoint_indices()
                    if self._all_points_scanned():
                        self.log.debug("Scan is over.")
                        self.stop_scan()
                    elif not self.is_scan_running:
                        self.log.debug("Scan was interrupted, not looping back.")
                        self.stop_scan()
                    else:
                        self._prepare_movement({
                            "grating" : self._grating_positions[self._grating_position_index],
                            "tension" : self._tension_positions[self._tension_position_index],
                        })
                        self.__start_write_timer()
                        self.sigNextDataChunk.emit()
        except:
            self.log.exception("")
            self.stop_scan()

    # ScanningProbeInterface
    def get_constraints(self):
        return self.constraints
    def reset(self):
        pass
    def configure_scan(self, scan_settings):
        self.log.debug(f"configuring scan with settings {scan_settings}")
        if self.is_scan_running:
            self.log.error("Cannot configure scan parameters while a scan is "
                           "running. Stop the current scan and try again.")
            return True, self.scan_settings
        axes = scan_settings.get('axes', self._current_scan_axes)
        ranges = tuple(
            (min(r), max(r)) for r in scan_settings.get('range', self._current_scan_ranges)
        )
        resolution = scan_settings.get('resolution', self._current_scan_resolution)
        frequency = float(scan_settings.get('frequency', self._current_scan_frequency))
        if not set(axes).issubset(["tension", "grating"]):
            self.log.error('Unknown axes names encountered. Valid axes are "tension" and "grating".')
            return True, self.scan_settings
        if len(axes) != len(ranges) or len(axes) != len(resolution):
            self.log.error('"axes", "range" and "resolution" must have same length.')
            return True, self.scan_settings
        for i, ax in enumerate(axes):
            for axis_constr in self.constraints.axes.values():
                if ax == axis_constr.name:
                    break
            if ranges[i][0] < axis_constr.min_value or ranges[i][1] > axis_constr.max_value:
                self.log.error('Scan range out of bounds for axis "{0}". Maximum possible range'
                               ' is: {1}'.format(ax, axis_constr.value_range))
                return True, self.scan_settings
            if resolution[i] < axis_constr.min_resolution or resolution[i] > axis_constr.max_resolution:
                self.log.error('Scan resolution out of bounds for axis "{0}". Maximum possible '
                               'range is: {1}'.format(ax, axis_constr.resolution_range))
                return True, self.scan_settings
            if i == 0:
                if frequency < axis_constr.min_frequency or frequency > axis_constr.max_frequency:
                    self.log.error('Scan frequency out of bounds for fast axis "{0}". Maximum '
                                   'possible range is: {1}'
                                   ''.format(ax, axis_constr.frequency_range))
                    return True, self.scan_settings
        self.log.info("Waiting for lock.")
        with self._thread_lock_data:
            try:
                self._scan_data = self._create_scan_data(axes, ranges, resolution, frequency)
                self._tension_positions,self._grating_positions = self._init_scan_arrays(self._scan_data)
                self._grating_position_index = 0
                self._tension_position_index = 0
            except:
                self.log.exception("")
                return True, self.scan_settings

        try:
            self.log.debug("It is time to configure the NI finite sampling hardware.")
            self.counter().set_sample_rate(frequency)
            self.counter().set_active_channels((self._counter_apd_channel,))
        except:
            self.log.exception("")
            return True, self.scan_settings

        self._current_scan_resolution = tuple(resolution)
        self._current_scan_ranges = ranges
        self._current_scan_axes = tuple(axes)
        self._current_scan_frequency = frequency

        self.log.debug("configured!")

        return False, self.scan_settings

    def move_absolute(self, position, velocity=None, blocking=False):
        if self.is_scan_running:
            self.log.error('Cannot move the scanner while, scan is running')
            return self.get_target()
        if not set(position).issubset(self.get_constraints().axes):
            self.log.error('Invalid axes name in position')
            return self.get_target()
        try:
            self._prepare_movement(position, velocity=velocity)
            self.log.debug("Setting setpoint.")
            self.__start_write_timer()
            if blocking:
                self.__wait_on_move_done()


            return self.get_target()
        except:
            self.log.exception("Couldn't move: ")

    def move_relative(self, position, velocity=None, blocking=False):
        """ Move the scanning probe by a relative distance from the current target position as fast
        as possible or with a defined velocity.

        Log error and return current target position if something fails or a 1D/2D scan is in
        progress.
        """
        current_position = self.bare_scanner.get_position()
        end_pos = {ax: current_position[ax] + distance[ax] for ax in distance}
        self.move_absolute(end_pos, velocity=velocity, blocking=blocking)

        return end_pos
    def get_target(self):
        if self.is_scan_running:
            return self._stored_target_pos
        else:
            return self._target_pos
    def get_position(self):
        """ Get a snapshot of the actual scanner position (i.e. from position feedback sensors).
        For the same target this value can fluctuate according to the scanners positioning accuracy.

        For scanning devices that do not have position feedback sensors, simply return the target
        position (see also: ScanningProbeInterface.get_target).

        @return dict: current position per axis.
        """
        with self._thread_lock_cursor:
            tension_position = self.fzw_process_control().get_setpoint("tension")
            grating_position = self.cem().get_process_value("grating")
        pos = {
            "tension": tension_position,
            "grating": grating_position
        }
        return pos
        
    def start_scan(self):
        try:
            self.log.debug("start_scan")
            #self.log.debug(f"Start scan in thread {self.thread()}, QT.QThread {QtCore.QThread.currentThread()}... ")
            if self.thread() is not QtCore.QThread.currentThread():
                QtCore.QMetaObject.invokeMethod(self, '_start_scan',
                                                QtCore.Qt.BlockingQueuedConnection)
            else:
                self._start_scan()
        except:
            self.log.exception("")
            return -1
        return 0

    @QtCore.Slot()
    def _start_scan(self):
        """

        @return (bool): Failure indicator (fail=True)
        """
        try:
            self.log.debug("scanning requested.")
            if self._scan_data is None:
                # todo: raising would be better, but from this delegated thread exceptions get lost
                self.log.error('Scan Data is None. Scan settings need to be configured before starting')

            if self.is_scan_running:
                self.log.error('Cannot start a scan while scanning probe is already running')

            with self._thread_lock_data:
                self._scan_data.new_scan()
                #self.log.debug(f"New scan data: {self._scan_data.data}, position {self._scan_data._position_data}")
                self._stored_target_pos = self.get_target().copy()
                self.log.debug(f"Target pos at scan start: {self._stored_target_pos}")
                self._scan_data.scanner_target_at_start = self._stored_target_pos

            # todo: scanning_probe_logic exits when scanner not locked right away
            # should rather ignore/wait until real hw timed scanning starts
            self.module_state.lock()

            first_scan_position = {ax: pos[0] for ax, pos
                                   in zip(self.scan_settings['axes'], self.scan_settings['range'])}
            self.log.debug("calling move and start.")
            self._move_to_and_start_scan(first_scan_position)
        except Exception:
            self.module_state.unlock()
            self.log.exception("Starting scan failed: ")
    def stop_scan(self):
        """
        @return bool: Failure indicator (fail=True)
        # todo: return values as error codes are deprecated
        """
        if self.thread() is not QtCore.QThread.currentThread():
            QtCore.QMetaObject.invokeMethod(self, '_stop_scan',
                                            QtCore.Qt.BlockingQueuedConnection)
        else:
            self._stop_scan()

        return 0

    @QtCore.Slot()
    def _stop_scan(self):
        # self.log.debug("Stopping scan...")
        self._start_scan_after_cursor = False  # Ensure Scan HW is not started after movement
        self._abort_cursor_movement()
        if self.counter().module_state() == "locked":
            self.fzw_sampling().stop_buffered_acquisition()
        if self.module_state() == "locked":
            self.module_state.unlock()
        # self.log.debug("Module unlocked")
        self.log.debug(f"Finished scan, move to stored target: {self._stored_target_pos}")
        self.move_absolute(self._stored_target_pos)
        self._stored_target_pos = dict()
        
    def get_scan_data(self):
        """
        @return (ScanData): ScanData instance used in the scan
        """
        if self._scan_data is None:
            raise RuntimeError('ScanData is not yet configured, please call "configure_scan" first')
        try:
            with self._thread_lock_data:
                return self._scan_data.copy()
        except:
            self.log.exception("")
            
    def emergency_stop(self):
        # There's never an emergency.
        pass
        
