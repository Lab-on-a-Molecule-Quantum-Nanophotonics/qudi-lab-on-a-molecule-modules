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

import serial
import numpy as np

from qudi.core.configoption import ConfigOption
# from qudi.interface.scanning_laser_interface import ScanningLaserInterface, ScanningState, ScanningLaserReturnError
from qudi.interface.data_instream_interface import DataInStreamInterface, DataInStreamConstraints, SampleTiming, StreamingMode, ScalarConstraint
from qudi.interface.process_control_interface import ProcessControlConstraints, ProcessControlInterface
from qudi.interface.switch_interface import SwitchInterface
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from qudi.util.mutex import Mutex
from qudi.util.overload import OverloadedAttribute

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
                "HV,MOD":("EXT", "RAMP")
        }
    def get_state(self, switch):
        return self._mod_status()
    def set_state(self, switch, state):
        self._set_mod_status(state)

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
        

class MOGLabsLaser(DataInStreamInterface, FiniteSamplingInputInterface):
    """A class to control our MOGLabs laser to perform excitation spectroscopy.

    Example config:

    """
    address = ConfigOption('address', "127.0.0.1")
    port = ConfigOption('port', 7805)
    timeout_seconds = ConfigOption('timeout_seconds', 0.2)
    set_mode = ConfigOption('set_mode', "fast")
    refresh_rate = ConfigOption('refresh_rate', 10)
    _grating_steps = ConfigOption('grating_steps', 100)

    sig_connect_cem = QtCore.Signal()
    sig_scan_done = QtCore.Signal()
    _sig_next_frame = QtCore.Signal()
    _sig_send_data = QtCore.Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraints: Optional[DataInStreamConstraints] = None
        #self.socket_thread = QtCore.QThread()
        #self._socket_handler = SocketHandler(self)
        self._lock = Mutex()
        self._data_buffer: Optional[np.ndarray] = None
        self._timestamp_buffer: Optional[np.ndarray] = None
        self._current_buffer_position = 0
        self._time_offset = 0
        self._last_read = 0
        self._frame_size = 1
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.timeouted = False

    # Qudi activation / deactivation
    def on_activate(self):
        """Activate module.
        """
        self.log.info("Starting the MOGLabs laser.")
        #self._socket_handler.moveToThread(self.socket_thread)
        #self.response_queue = queue.Queue()
        #self.sig_connect_cem.connect(self._socket_handler.run, QtCore.Qt.QueuedConnection)
        #self._socket_handler.response_received.connect(self.on_receive, QtCore.Qt.QueuedConnection)
        #self._sig_send_data.connect(self._socket_handler.send, QtCore.Qt.QueuedConnection)
        # Enable external modulation once connected to the interface.
        #self._socket_handler.connection_established.connect(self._external_modulation, QtCore.Qt.QueuedConnection)
        #self.socket_thread.start()
        #self.sig_connect_cem.emit()
        self._time_offset = 0

        self._constraints = DataInStreamConstraints(
            channel_units = {
                'wavelength': 'nm',
            },
            sample_timing=SampleTiming.TIMESTAMP,
            streaming_modes = [StreamingMode.CONTINUOUS],
            data_type=np.float64,
            channel_buffer_size=ScalarConstraint(default=2**16,
                                                 bounds=(2, (2**32)//10),
                                                 increment=1,
                                                 enforce_int=True),
            sample_rate=ScalarConstraint(default=2,bounds=(2,2),increment=0),
        )
        self._constraints_finite_sampling = FiniteSamplingInputConstraints(
            channel_units = {
                'wavelength': 'm',
            },
            frame_size_limits=(1,2**16),
            sample_rate_limits=(1,1e3),
        )
        self.log.info("Connecting to MOGLabs interface.")
        self.socket.connect((self.address, self.port))
        self.timeouted = False
        self.socket.settimeout(1)
        self._external_modulation()

    def on_deactivate(self):
        """Deactivate module.
        """
        self._external_modulation(False)
        #self.socket_thread.quit()
        #self._socket_handler.response_received.disconnect()
        #self.sig_connect_cem.disconnect()
        self.log.info("Disconnecting from MOGLabs interface.")
        self.socket.close()
        self.stop_stream()
        self._data_buffer = None
        self._timestamp_buffer = None

    # Internal communication facilities
    def send_and_recv(self, value, timeout_s=1, check_ok=True):
        if self.timeouted:
            self.timeouted = False
            self.socket.settimeout(0)
            chunk = self.socket.recv(1024)
            while chunk is not None:
                chunk = self.socket.recv(1024)
            
        self.socket.settimeout(timeout_s)
        self.socket.sendall(value.encode("utf8"))
        recv = None
        try:
            recv = self.socket.recv(1024)
        except TimeoutError:
            self.log.warning("Timeout while waiting for a response to %s", value)
            self.timeouted = True
        if check_ok:
            if recv is None:
                self.log.error(f"Command \"{value}\" did not return.")
            elif not recv.decode("utf8").startswith("OK"):
                self.log.error(f"Command \"{value}\" errored: \"{recv}\"")
                recv = None
        return recv
            
    def _external_modulation(self, on=True):
        if on:
            self.log.info("Enabling external modulation for the piezo.")
            self.send_and_recv("mld,hv,mod,ext", timeout_s=2)
        else:
            self.log.info("Disabling external modulation for the piezo.")
            self.send_and_recv("mld,hv,mod,ramp", timeout_s=2)

    def _set_continuous_acquisition(self):
        self.log.info("Disabling external triggering of the wavemeter.")
        ret = self.send_and_recv("fzw,meas,extrig,off")
        if ret is None:
            return
        self.send_and_recv("fzw,meas,clear")

    def _set_oneshot_acquisition(self):
        self.log.info("Enabling external triggering of the wavemeter.")
        ret = self.send_and_recv("fzw,meas,extrig,on")
        if ret is None:
            return
        self.send_and_recv("fzw,meas,clear")

    def _dump_data(self, buffer_data : Optional[np.ndarray] = None , buffer_timestamp : Optional[np.ndarray] = None, timeout=0.5) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.timeouted:
            self.timeouted = False
            self.socket.settimeout(0)
            chunk = self.socket.recv(1)
            while chunk is not None:
                chunk = self.socket.recv(1)
        self.socket.settimeout(timeout)
        self.socket.sendall(b"fzw,meas,dump")
        recv = b''
        while True:
            try:
                recv += self.socket.recv(1024)
            except TimeoutError:
                self.log.warning("Timeout while waiting for dump response")
                self.timeouted = True
                break
            if recv[-2:] == b'\r\n':
                break
        if len(recv) < 0:
            return np.empty(0),np.empty(0),0
        try:
            recv = eval(recv)
        except:
            self.log.warning("Could not parse the response.")
            return np.empty(0),np.empty(0),0

        size = struct.unpack("<I", recv[:4])[0]
        number_of_measurements = size//10
        if buffer_data is None:
            buffer_data = np.empty(number_of_measurements, dtype=np.float64)
        if buffer_timestamp is None:
            buffer_timestamp = np.empty(number_of_measurements, dtype=np.float64)
        dump_stop = min(number_of_measurements, len(buffer_timestamp))
        
        recv = recv[4:]

        dump_index = 0
        remaining_bytes = size
        d,_ = divmod(len(recv),10)
        for i in range(0,d*10,10):
            time,wavelength,_,_,_,_=struct.unpack("<HIbbbb", recv[i:(i+10)])
            remaining_bytes -= 10
            if dump_index < dump_stop:
                buffer_timestamp[dump_index] = time / 1000
                buffer_data[dump_index] = (wavelength * 1200.0 / (2**32 - 1)) * 1e-9
                dump_index += 1

        if buffer_timestamp.max() < self._time_offset:
            buffer_timestamp = buffer_timestamp + self._time_offset
        self._time_offset += buffer_timestamp.max()
        perm = np.argsort(buffer_timestamp)

        return buffer_data[perm],buffer_timestamp[perm],dump_index+1
        
    def _fzw_pid_enabled():
        r = self.send_and_recv("fzw,pid,status")
        return r == "ENABLED"
        
    def _set_fzw_pid_enabled(state : bool):
        if state:
            self.send_and_recv("fzw,pid,enable")
        else:
            self.send_and_recv("fzw,pid,disable")
            
    def _mod_status(self):
        return self.send_and_recv("mld,hv,mod", check_ok=False).decode("utf8").rstrip()
        
    def _set_mod_status(self, val):
        return self.send_and_recv(f"mld,hv,mod,{val}")
        
    def _motor_range(self):
        mini,maxi = self.send_and_recv("cem,motor,travel", check_ok=False).decode("utf8").split(" ")
        return int(mini), int(maxi)
    
    def _motor_position(self):
        return int(self.send_and_recv("cem,motor,position", check_ok=False))
        
    def _set_motor_position(self, value):
        return self.send_and_recv(f"cem,motor,dest,{value}")
        
    def _move_motor_rel(self, value):
        return self.send_and_recv(f"cem,motor,step,{value}")
        
    def _frequency(self):
        f = self.send_and_recv("freq", check_ok=False)
        if f is None:
            return np.empty(0)
        try:
            return np.array(float(f)),None 
        except ValueError:
            return np.empty(0), None
        
    def _calibration_scan(self):    
        mod_status = self._mod_status()
        self._set_mod_status("none")
        mini,maxi = self._motor_range()
        self._motor_positions = np.arange(start=mini, stop=maxi, step=500, dtype=int)
        self._frequencies = np.empty(len(self._motor_positions), dtype=np.float64)
        self.log.info("Starting calibration scan...")
        for i in range(len(self._motor_positions)):
            self._set_motor_position(self._motor_positions[i])
            while self._motor_position() != self._motor_positions[i]:
                time.sleep(0.1)
            self.send_and_recv("fzw,meas,softrig")
            f = self.send_and_recv("freq", check_ok=False)
            if f is None:
                self._frequencies[i] = np.nan
            try:
                self._frequencies[i] = float(f) 
            except ValueError:
                self._frequencies[i] = np.nan
        self._motor_positions = self._motor_positions[self._frequencies != np.nan]
        self._frequencies = self._frequencies[self._frequencies != np.nan]
        self._set_mod_status(mod_status)        
        self.log.info("Calibration scan done.")
        
    # DataInStreamInterface
    constraints = OverloadedAttribute()
    @constraints.overload("DataInStreamInterface")
    @property
    def constraints(self) -> DataInStreamConstraints:
        if self._constraints is None:
            raise ValueError("Constraints have not yet been initialized.")
        return self._constraints

    def start_stream(self) -> None:
        """ Start the data acquisition/streaming """
        with self._lock:
            if self.module_state() == 'idle':
                self.module_state.lock()
                self._set_continuous_acquisition()
            else:
                self.log.warning('Unable to start input stream. It is already running.')

    def stop_stream(self) -> None:
        """ Stop the data acquisition/streaming """
        with self._lock:
            if self.module_state() == 'locked':
                self.module_state.unlock()
            else:
                self.log.warning('Unable to stop wavemeter input stream as nothing is running.')

    def read_data_into_buffer(self,
                              data_buffer: np.ndarray,
                              samples_per_channel: int,
                              timestamp_buffer: Optional[np.ndarray] = None) -> None:
        """ Read data from the stream buffer into a 1D numpy array given as parameter.
        Samples of all channels are stored interleaved in contiguous memory.
        In case of a multidimensional buffer array, this buffer will be flattened before written
        into.
        The 1D data_buffer can be unraveled into channel and sample indexing with:

            data_buffer.reshape([<samples_per_channel>, <channel_count>])

        The data_buffer array must have the same data type as self.constraints.data_type.

        In case of SampleTiming.TIMESTAMP a 1D numpy.float64 timestamp_buffer array has to be
        provided to be filled with timestamps corresponding to the data_buffer array. It must be
        able to hold at least <samples_per_channel> items:

        If the number of available samples is too low, you will get zeros at the end.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            self._dump_data(data_buffer, timestamp_buffer)

    def read_available_data_into_buffer(self,
                                        data_buffer: np.ndarray,
                                        timestamp_buffer: Optional[np.ndarray] = None) -> int:
        """ Read data from the stream buffer into a 1D numpy array given as parameter.
        All samples for each channel are stored in consecutive blocks one after the other.
        The number of samples read per channel is returned and can be used to slice out valid data
        from the buffer arrays like:

            valid_data = data_buffer[:<channel_count> * <return_value>]
            valid_timestamps = timestamp_buffer[:<return_value>]

        See "read_data_into_buffer" documentation for more details.

        This method will read all currently available samples into buffer. If number of available
        samples exceeds buffer size, read only as many samples as fit into the buffer and drop the rest.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            if timestamp_buffer is None:
                raise RuntimeError(
                    'SampleTiming.TIMESTAMP mode requires a timestamp buffer array'
                )

            return self._dump_data(data_buffer, timestamp_buffer)[-1]

    def read_data(self,
                  samples_per_channel: Optional[int] = None
                  ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """ Read data from the stream buffer into a 1D numpy array and return it.
        All samples for each channel are stored in consecutive blocks one after the other.
        The returned data_buffer can be unraveled into channel samples with:

            data_buffer.reshape([<samples_per_channel>, <channel_count>])

        The numpy array data type is the one defined in self.constraints.data_type.

        In case of SampleTiming.TIMESTAMP a 1D numpy.float64 timestamp_buffer array will be
        returned as well with timestamps corresponding to the data_buffer array.

        If samples_per_channel is omitted all currently available samples are read from buffer.
        This method will not return until all requested samples have been read or a timeout occurs.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')

            buffer_data, buffer_timestamp, l = self._dump_data()
            if samples_per_channel is not None and l > samples_per_channel:
                buffer_data = buffer_data[:samples_per_channel]
                buffer_timestamp = buffer_timestamp[:samples_per_channel]
            return buffer_data, buffer_timestamp

    def read_single_point(self) -> Tuple[np.ndarray, Union[None, np.float64]]:
        """ This method will initiate a single sample read on each configured data channel.
        In general this sample may not be acquired simultaneous for all channels and timing in
        general can not be assured. Us this method if you want to have a non-timing-critical
        snapshot of your current data channel input.
        May not be available for all devices.
        The returned 1D numpy array will contain one sample for each channel.

        In case of SampleTiming.TIMESTAMP a single numpy.float64 timestamp value will be returned
        as well.
        """
        with self._lock:
            if self.module_state() != 'locked':
                raise RuntimeError('Unable to read data. Stream is not running.')
            self.send_and_recv("fzw,meas,softrig")
            f = self.send_and_recv("freq", check_ok=False)
            if f is None:
                return np.empty(0),None
            try:
                return np.array(float(f)),None 
            except ValueError:
                return np.empty(0), None

    @property
    def available_samples(self) -> int:
        # self.log.warning("Available samples reading is not available on the MOGLabs FZW.")
        return 1

    @property
    def sample_rate(self) -> float:
        # self.log.warning("Sample rate reading is not available on the MOGLabs FZW.")
        return 1e3

    @property
    def channel_buffer_size(self) -> int:
        return (2**32)/10

    @property 
    def streaming_mode(self) -> StreamingMode:
        return StreamingMode.CONTINUOUS

    @property
    def active_channels(self):
        return ['wavelength']

    def configure(self,
                  active_channels: Sequence[str],
                  streaming_mode: Union[StreamingMode, int],
                  channel_buffer_size: int,
                  sample_rate: float) -> None:
        """ Configure a data stream. See read-only properties for information on each parameter. """
        # self.log.warning("There is nothing to configure on the MOGLabs FZW!")

    # FiniteSamplingInputInterface
    @constraints.overload("FiniteSamplingInputInterface")
    @property
    def constraints(self):
        return self._constraints_finite_sampling

    @property 
    def frame_size(self):
        return self._frame_size

    @property
    def samples_in_buffer(self):
        return 1
    
    def set_active_channels(self, _):
        pass 

    def set_sample_rate(self, rate):
        pass

    def set_frame_size(self, size):
        self._frame_size = size

    def start_buffered_acquisition(self):
        self._set_oneshot_acquisition()

    def stop_buffered_acquisition(self):
        self._set_continuous_acquisition()

    def get_buffered_samples(self, number_of_samples: Optional[int] =None):
        with self._lock:
            if number_of_samples is None:
                buffer_data, _, l = self._dump_data(timeout=3)
                return {"wavelength": buffer_data}
            elif number_of_samples > self.frame_size:
                raise ValueError(f"You are asking for too many samples ({number_of_samples} for a maximum of {self.frame_size}.")
            else:
                buffer_data, _, l = self._dump_data(timeout=3)
                while l < number_of_samples:
                    time.sleep(0.1)
                    added_buffer_data, added_buffer_timestamp, added_l = self._dump_data(timeout=3)
                    l += added_l
                    buffer_data = np.concatenate(buffer_data, added_buffer_data)
                return {"wavelength": buffer_data[:number_of_samples]}

    def acquire_frame(self, frame_size=None):
        old_frame_size = self.frame_size
        if frame_size is not None:
            self.set_frame_size(frame_size)
        self.start_buffered_acquisition()
        data = self.get_buffered_samples(self.frame_size)
        self.stop_buffered_acquisition()
        if frame_size is not None:
            self.set_frame_size(old_frame_size)
        return data

    # MotorInterface
    
    def get_constraints(self):
        axis0 = {}
        axis0['label'] = 'grating'
        axis0['unit'] = None
        axis0['ramp'] = None
        mini,maxi = self._motor_range()
        axis0['pos_min'] = mini
        axis0['pos_max'] = maxi
        axis0['pos_step'] = self._grating_steps
        axis0['vel_min'] = None
        axis0['vel_max'] = None
        axis0['vel_step'] = None
        axis0['acc_min'] = None
        axis0['acc_max'] = None
        axis0['acc_step'] = None
        constraints = {}
        constraints[axis0['label']] = axis0
        return constraints
        
    def move_rel(self,  param_dict):
        rel = param_dict['grating']
        self._move_motor_rel(rel)
        
    def move_abs(self, param_dict):
        rel = param_dict['grating']
        self._set_motor_position(rel)
        
    def abort(self):
        pass
        
    def get_pos(self, param_list=None):
        return {'grating': self._motor_position()}
        
    def get_status(self, param_list=None):
        pass
        
    def calibrate(self, param_list=None):
        pass
        
    def get_velocity(self, param_list=None):
        pass
        
    def set_velocity(self, param_dict):
        pass
