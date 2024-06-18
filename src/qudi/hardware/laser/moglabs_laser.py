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
# from qudi.interface.scanning_laser_interface import ScanningLaserInterface, ScanningState, ScanningLaserReturnError
from qudi.interface.data_instream_interface import DataInStreamInterface, DataInStreamConstraints, SampleTiming, StreamingMode, ScalarConstraint
from qudi.interface.process_control_interface import ProcessControlConstraints, ProcessControlInterface
from qudi.interface.switch_interface import SwitchInterface
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from qudi.interface.scanning_probe_interface import ScanningProbeInterface, ScanData, ScannerChannel, ScannerAxis, ScanConstraints
from qudi.util.mutex import Mutex
from qudi.core.connector import Connector
from qudi.util.overload import OverloadedAttribute
from qudi.util.helpers import in_range
from qudi.util.enums import SamplingOutputMode

class MOGLABSMotorizedLaserDriver(SwitchInterface):
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
        self.serial.baudrate=115200
        self.serial.bytesize=8
        self.serial.parity='N'
        self.serial.stopbits=1
        self.serial.timeout=1
        self.serial.writeTimeout=0
        self.serial.port=self.port
        self.serial.open()

    def on_deactivate(self):
        """Deactivate module.
        """
        self.serial.close()

    @property
    def name(self):
        return "LDD"
    @property
    def available_states(self):
        return {
                "HV,MOD":("EXT", "RAMP"),
                "Temp. control":("OFF", "ON"),
                "Curr. control":("OFF", "ON"),
        }
    def get_state(self, switch):
        if switch == "HV,MOD":
            return self._mod_status()
        elif switch == "Temp. control":
            return self._temp_status()
        else:
            return self._current_status()

    def set_state(self, switch, state):
        if switch == "HV,MOD":
            self._set_mod_status(state)
        elif switch == "Temp. control":
            self._set_temp_status(state)
        else:
            self._set_current_status(state)

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

class MOGLABSCateyeLaser(ProcessControlInterface):
    port = ConfigOption("port")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serial = serial.Serial()

    # Qudi activation / deactivation
    def on_activate(self):
        """Activate module.
        """
        self.serial.baudrate=115200
        self.serial.bytesize=8
        self.serial.parity='N'
        self.serial.stopbits=1
        self.serial.timeout=1
        self.serial.writeTimeout=0
        self.serial.port=self.port
        self.serial.open()

    def on_deactivate(self):
        """Deactivate module.
        """
        self.serial.close()
    def set_setpoint(self, channel, value):
        self._set_motor_position(value)

    def get_setpoint(self, channel):
        return self._get_motor_setpoint()

    def get_process_value(self, channel):
        return self._motor_position()

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
        return ProcessControlConstraints(
            ["grating"],
            ["grating"],
            {"grating":"step"},
            {"grating":self._motor_range()},
            {"grating":int},
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

    def _motor_range(self):
        mini,maxi = self.send_and_recv("motor,travel", check_ok=False).split(" ")
        return int(mini), int(maxi)
    
    def _motor_position(self):
        return int(self.send_and_recv("motor,position", check_ok=False))
        
    def _set_motor_position(self, value):
        return self.send_and_recv(f"motor,dest,{value}")
        
    def _get_motor_setpoint(self):
        return int(self.send_and_recv(f"motor,dest", check_ok=False))
        
    def _move_motor_rel(self, value):
        return self.send_and_recv(f"motor,step,{value}")

    def _motor_status(self):
        return self.send_and_recv("motor,status").rstrip()
        

class MOGLabsConfocalScanningLaserInterfuse(ScanningProbeInterface):
    """A class to control our MOGLabs laser to perform excitation spectroscopy.

    Example config:

    """
    cem = Connector(name="cateye_laser", interface="ProcessControlInterface")
    ldd = Connector(name="laser_driver", interface="SwitchInterface")
    fzw_sampling = Connector(name="wavemeter_sampling", interface="FiniteSamplingInputInterface")
    fzw_switch = Connector(name="wavemeter_switch", interface="SwitchInterface")
    ni_sampling = Connector(name="dac", interface="FiniteSamplingIOInterface")
    ni_ao = Connector(name='analog_output', interface='ProcessSetpointInterface')

    _ni_piezo_channel = ConfigOption(name="ni_piezo_channel", missing="error")
    _ni_apd_channel = ConfigOption(name="ni_apd_channel", missing="error")

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
        self._scanner_distance_atol = 1e-9
        self._start_scan_after_cursor = False
        self._abort_cursor_move = True
        self.__t_last_follow = None
        
        self._grating_positions = None
        self._is_moving_grating = False
        self._grating_position_index = 0

        self.__ni_ao_write_timer = None
        self._thread_lock_cursor = Mutex()
        self._thread_lock_data = Mutex()

        self._last_timestamp_finite_sampling = None

    def on_activate(self):
        self.axes = [
            ScannerAxis(
                name="grating",
                unit="step",
                value_range=self.cem().constraints.channel_limits["grating"],
                step_range=(0, 200),
                resolution_range=(1,10000),
            ),
            ScannerAxis(
                name="piezo",
                unit="V",
                value_range=(-10,10),
                step_range=(0, 10),
                resolution_range=(1, 1e9),
            ),
        ]
        self.channels = [
            ScannerChannel(
                name="APD",
                unit="c/s",
            ),
            ScannerChannel(
                name="wavelength",
                unit="m",
            )
        ]
        self.constraints = ScanConstraints(
            axes=self.axes,
            channels=self.channels,
            backscan_configurable=False,
            has_position_feedback=True,
            square_px_only=False,
        )

        self._target_pos = self.get_position()  
        self._toggle_piezo_setpoint_channel(False)  # And free ao resources after that
        self._t_last_move = time.perf_counter()
        self.__init_ao_timer()
        self.sigNextDataChunk.connect(self._fetch_data_chunk, QtCore.Qt.QueuedConnection)

    def on_deactivate(self):
        self._abort_cursor_movement()
        if self.ni_sampling().is_running:
            self.ni_sampling().stop_buffered_frame()

    # Internal facilities
    def _toggle_piezo_setpoint_channel(self, enable):
        ni_ao = self.ni_ao()
        ni_ao.set_activity_state(self._ni_piezo_channel, enable)

    @property
    def _piezo_setpoint_channel_active(self) -> bool:
        return self.ni_ao().activity_states[self._ni_piezo_channel]

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
                ni_scan_dict,grating_scan_array = self._init_scan_arrays(scan_data)
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
        scan_voltages = {self._ni_piezo_channel: scan_coords["piezo"]}

        grating_positions = scan_coords.get("grating", [self.get_position()["grating"]])

        return scan_voltages, grating_positions

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
            if not self._piezo_setpoint_channel_active:
                self._toggle_piezo_setpoint_channel(True)

            constr = self.get_constraints()

            for axis, pos in position.items():
                in_range_flag, _ = in_range(pos, *constr.axes[axis].value_range)
                if not in_range_flag:
                    position[axis] = float(constr.axes[axis].clip_value(position[axis]))
                    self.log.warning(f'Position {pos} out of range {constr.axes[axis].value_range} '
                                     f'for axis {axis}. Value clipped to {position[axis]}')
                # TODO Adapt interface to use "in_range"?
                self._target_pos[axis] = position[axis]

            #self.log.debug(f"New target pos: {self._target_pos}")

            # TODO Add max velocity as a hardware constraint/ Calculate from scan_freq etc?
            # if velocity is None:
            #     velocity = self.__max_move_velocity
            # v_in_range, velocity = in_range(velocity, 0, self.__max_move_velocity)
            # if not v_in_range:
            #     self.log.warning(f'Requested velocity is exceeding the maximum velocity of {self.__max_move_velocity} '
            #                      f'm/s. Move will be done at maximum velocity')
            #
            # self._follow_velocity = velocity

        #self.log.debug("Movement prepared")
        # TODO Keep other axis constant?

    def __init_ao_timer(self):
        self.__ni_ao_write_timer = QtCore.QTimer(parent=self)

        self.__ni_ao_write_timer.setSingleShot(True)
        self.__ni_ao_write_timer.timeout.connect(self.__ao_cursor_write_loop, QtCore.Qt.QueuedConnection)
        self.__ni_ao_write_timer.setInterval(1e3*self._min_step_interval)  # (ms), dynamically calculated during write loop

    def __start_ao_write_timer(self):
        #self.log.debug(f"ao start write timer in thread {self.thread()}, QT.QThread {QtCore.QThread.currentThread()} ")
        try:
            if not self.is_move_running:
                #self.log.debug("Starting AO write timer...")
                if self.thread() is not QtCore.QThread.currentThread():
                    QtCore.QMetaObject.invokeMethod(self.__ni_ao_write_timer,
                                                    'start',
                                                    QtCore.Qt.BlockingQueuedConnection)
                else:
                    self.__ni_ao_write_timer.start()
            else:
                pass
                #self.log.debug("Dropping timer start, already running")

        except:
            self.log.exception("")

    def __ao_cursor_write_loop(self):

        t_start = time.perf_counter()
        try:
            current_pos_vec = self._pos_dict_to_vec(self.get_position())
            self.log.debug("loooooping")

            with self._thread_lock_cursor:
                self.log.debug("Got lock")
                stop_loop = self._abort_cursor_move
                self.log.debug(f"Abort is {stop_loop}")
                target_pos_vec = self._pos_dict_to_vec(self._target_pos)
                connecting_vec = target_pos_vec - current_pos_vec
                distance_to_target = np.linalg.norm(connecting_vec)
                self.log.debug(f"distance to target {distance_to_target}.")

                # Terminate follow loop if target is reached
                if distance_to_target < self._scanner_distance_atol:
                    stop_loop = True

                self.log.debug(f"stop_loop is {stop_loop}")

                if not stop_loop:
                    if not self.__t_last_follow:
                        self.__t_last_follow = time.perf_counter()
                    if not self._piezo_setpoint_channel_active:
                        self._toggle_piezo_setpoint_channel(True)
                    self.log.debug("Setting setpoint.")
                    self.ni_ao().set_setpoint(self._ni_piezo_channel, self._target_pos["piezo"])
                    self.cem().set_setpoint("grating", self._target_pos["grating"])
                    self.log.debug("Time to stqrt the timer.")
                    self.__ni_ao_write_timer.start(int(round(1000 * self._min_step_interval)))
            if stop_loop:
                #self.log.debug(f'Cursor_write_loop stopping at {current_pos_vec}, dist= {distance_to_target}')
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
        Abort the movement and release ni_ao resources.
        """

        #self.log.debug(f"Aborting move.")
        self._target_pos = self.get_position()

        with self._thread_lock_cursor:
            self._abort_cursor_move = True
            self.__t_last_follow = None
            self._toggle_piezo_setpoint_channel(False)

    def _move_to_and_start_scan(self, position):
        self._prepare_movement(position)
        self._start_scan_after_cursor = True
        #self.log.debug("Starting timer to move to scan position")
        self.__start_ao_write_timer()

    def _start_hw_timed_scan(self):
        try:
            self.log.debug("start hw scan.")
            self.ni_sampling().start_buffered_frame()
            self.fzw_sampling().start_buffered_acquisition()
            self._last_timestamp_finite_sampling = 0
            self.sigNextDataChunk.emit()
        except Exception as e:
            self.log.error(f'Could not start frame due to {str(e)}')
            self.module_state.unlock()

        self._start_scan_after_cursor = False

    def _current_data_index(self, chunk_size):
        scan_axes = set(self._scan_data.scan_axes)
        first_nan_idx = np.sum(~np.isnan(next(iter(self._scan_data.data.values()))))
        if len(scan_axes & {"grating", "piezo"}) == 2:
            return (slice(first_nan_idx, first_nan_idx+chunk_size), self._grating_position_index-1)
        elif len(scan_axes & {"grating", "piezo"}) == 1:
            return slice(first_nan_idx, first_nan_idx+chunk_size)
        else:
            raise ValueError(f"Inconsistent scan_axes for indexing: {scan_axes}")

    def _fetch_data_chunk(self):
        self.log.debug("fetch data chunk.")
        if self._is_moving_grating:
            self.log.debug("grating is moving")
            if self.cem().get_setpoint("grating") == self.cem().get_process_value("grating"):
                self._is_moving_grating = False
                self.ni_sampling().start_buffered_frame()
                self.fzw_sampling().start_buffered_acquisition()
            self.sigNextDataChunk.emit()
            return
        try:
            # self.log.debug(f'fetch chunk: {self._ni_finite_sampling_io().samples_in_buffer}, {self.is_scan_running}')
            # chunk_size = self._scan_data.scan_resolution[0] + self.__backwards_line_resolution
            chunk_size = 10  # TODO Hardcode or go line by line as commented out above?
            # Request a minimum of chunk_size samples per loop
            self.log.debug(f"There are {self.ni_sampling().samples_in_buffer} samples available in NI, and {self.fzw_sampling().samples_in_buffer} samples available in FZW")
            try:
                self.log.debug("Reading buffered samples for the NI")
                samples_dict_ni = self.ni_sampling().get_buffered_samples(chunk_size) \
                    if self.ni_sampling().samples_in_buffer < chunk_size\
                    else self.ni_sampling().get_buffered_samples()
            except ValueError:  # ValueError is raised, when more samples are requested then pending or still to get
                # after HW stopped
                samples_dict_ni = self.ni_sampling().get_buffered_samples()
            samples_apd = samples_dict_ni[self._ni_apd_channel]
            chunk_size = len(samples_apd)
            available_samples_fzw = self.fzw_sampling().samples_in_buffer
            chunk_size_reading = min(chunk_size, available_samples_fzw)
            complete_with_nan = chunk_size - chunk_size_reading
            self.log.debug(f"I will try to read {chunk_size_reading} samples from fzw and i will complete with {complete_with_nan} nan.")
            try:
                self.log.debug("Reading buffered samples for the FZW")
                samples_dict_fzw = self.fzw_sampling().get_buffered_samples(chunk_size)
            except ValueError:  # ValueError is raised, when more samples are requested then pending or still to get
                samples_dict_fzw = self.fzw_sampling().get_buffered_samples()

            firstindex_fzw = 0
            timestamps = samples_dict_fzw["timestamp"]
            wavelengths = np.concatenate((samples_dict_fzw["wavelength"], np.full(complete_with_nan, np.nan)))
            # jump_dt = 1000 / self.ni_sampling().sample_rate / 2
            # while timestamps[firstindex_fzw] - self._last_timestamp_finite_sampling < jump_dt/2:
            #     firstindex_fzw += 1
            # timestamps = timestamps[firstindex_fzw:]
            # wavelengths = wavelengths[firstindex_fzw:]
            # down_sampled_timestamps = [timestamps[0]]
            # down_sampled_wavelengths = [wavelengths[0]]
            # for i in range(len(timestamps)):
            #     if timestamps[i] - down_sampled_timestamps[-1] > jump_dt:
            #         down_sampled_timestamps.append(timestamps[i])
            #         down_sampled_wavelengths.append(wavelengths[i])
            self._last_timestamp_finite_sampling = timestamps[-1]

            self.log.debug(f"Read {len(wavelengths)} wavelength and {len(samples_dict_ni[self._ni_apd_channel])} datapoints.")
            sample_size = len(samples_apd)
            data_index = self._current_data_index(sample_size)
            self.log.debug(f"data index is {data_index}")
            with self._thread_lock_data:
                self._scan_data.data["APD"][data_index] = samples_apd
                self._scan_data.data["wavelength"][data_index] = wavelengths

                first_nan_idx = np.sum(~np.isnan(next(iter(self._scan_data.data.values()))))
                line_finished = first_nan_idx == self._number_of_points_piezo
                end_reached = line_finished and self._grating_position_index == len(self._grating_positions)

                if end_reached:
                    self.log.debug("finished scan")
                    self.stop_scan()
                elif line_finished:
                    self.log.debug("finished line")
                    if self.ni_sampling().is_running:
                        self.ni_sampling().stop_buffered_frame()
                        # self.log.debug("Frame stopped")
                    if self.fzw_sampling().module_state() == "locked":
                        self.fzw_sampling().stop_buffered_acquisition()
                    self._is_moving_grating = True
                    self.cem().set_setpoint("grating", self._grating_positions[self._grating_position_index])
                    self._grating_position_index += 1
                    self.sigNextDataChunk.emit()
                elif not self.is_scan_running:
                    self.log.debug("scan is not running, I quit.")
                    return
                else:
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
        self.log.debug(f"configuring scan zith settings {scan_settings}")
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
        if not set(axes).issubset(["piezo", "grating"]):
            self.log.error('Unknown axes names encountered. Valid axes are "piezo" and "grating".')
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
                ni_scan_dict, grating_positions = self._init_scan_arrays(self._scan_data)
                self._grating_positions = grating_positions
                self._is_moving_grating = False
                self._grating_position_index = 1

            except:
                self.log.exception("")
                return True, self.scan_settings

        try:
            self.log.debug("It is time to configure the NI finite sampling hardware.")
            self.ni_sampling().set_sample_rate(frequency)
            self.ni_sampling().set_active_channels(
                input_channels=(self._ni_apd_channel,),
                output_channels=(self._ni_piezo_channel,)
            )

            self.ni_sampling().set_output_mode(SamplingOutputMode.JUMP_LIST)
            self.log.debug(f"ni scan dict is {ni_scan_dict}.")
            self.ni_sampling().set_frame_data(ni_scan_dict)
            self._number_of_points_piezo = len(ni_scan_dict[self._ni_piezo_channel])

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

            self.__start_ao_write_timer()
            if blocking:
                self.__wait_on_move_done()

            self._t_last_move = time.perf_counter()

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
            if not self._piezo_setpoint_channel_active:
                self._toggle_piezo_setpoint_channel(True)
            piezo_voltage = self.ni_ao().get_setpoint(self._ni_piezo_channel.lower())
            grating_position = self.cem().get_setpoint("grating")

            pos = {
                "piezo": piezo_voltage,
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
        if self._piezo_setpoint_channel_active:
            self._abort_cursor_movement()
            # self.log.debug("Move aborted")

        if self.ni_sampling().is_running:
            self.ni_sampling().stop_buffered_frame()
            # self.log.debug("Frame stopped")
        if self.fzw_sampling().module_state() == "locked":
            self.fzw_sampling().stop_buffered_acquisition()

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
        
