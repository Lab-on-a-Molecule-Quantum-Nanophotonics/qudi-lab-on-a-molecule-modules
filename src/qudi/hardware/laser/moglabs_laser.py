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
import json

from PySide2 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.interface.scanning_laser_interface import ScanningLaserInterface, ScanningState, ScanningLaserReturnError

class SocketHandler(QtCore.QObject):
    """Handler class for the socket that communicates with the MOGLabs software
    in a dedicated thread."""

    response_received = QtCore.Signal(str)

    def __init__(self, parentclass):
        super().__init__()
        # The settings are stored within the parent class
        self._parentclass = parentclass
        self.log = self._parentclass.log
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.input_queue = queue.Queue()

    def connect_socket(self):
        self.log.info("Connecting to MOGLabs interface.")
        self.socket.connect((self._parentclass.address, self._parentclass.port))
        self.socket.settimeout(0.1)

    def disconnect_socket(self):
        self.log.info("Disconnecting from MOGLabs interface.")
        self.socket.close()

    @QtCore.Slot(str)
    def send(self, value):
        self.input_queue.put(bytes(value, "utf8"))

    def run(self):
        self.connect_socket()
        value = None
        while True:
            if self._parentclass.module_state() != 'running':
                break
            try:
                value = self.input_queue.get(block=False)
                self.socket.sendall(value)
            except queue.Empty:
                pass

            try:
                value = self.socket.recv(1024)
                self.response_received.emit(value)
            except TimeoutError:
                pass
        self.disconnect_socket()

class MOGLabsLaser(ScanningLaserInterface):
    """A class to control our MOGLabs laser to perform excitation spectroscopy.

    Example config:

    """
    _address = ConfigOption('address', "127.0.0.1")
    _port = ConfigOption('port', 7805)
    _peak_width = ConfigOption('peak_width', 0.2)
    _timeout_seconds = ConfigOption('timeout_seconds', 0.2)
    _set_mode = ConfigOption('set_mode', "fast")

    sig_connect_cem = QtCore.Signal()
    sig_scan_done = QtCore.Signal()

    def on_activate(self):
        """Activate module.
        """
        self.log.info("Starting the MOGLabs laser.")
        self.socket_thread = QtCore.QThread()
        self._socket_handler = SocketHandler(self)
        self._socket_handler.moveToThread(self.socket_thread)
        self.response_queue = queue.Queue()

        self.sig_connect_cem.connect(self._socket_handler.run)
        self._socket_handler.response_received.connect(self.on_receive)
        self.socket_thread.start()

    def on_deactivate(self):
        """Deactivate module.
        """
        self.socket_thread.quit()
        self._socket_handler.response_received.disconnect()
        self.sig_connect_cem.disconnect()

    def on_receive(self, value):
        self.response_queue.put(value)

    def send_and_recv(self, value, timeout_s=1):
        self._socket_handler.send(value)
        value = None
        try:
            value = self.response_queue.get(timeout=timeout_s)
        except TimeoutError:
            self.log.error("Timeout while waiting for a response to %s", value)
        return value

    def empty_input_queue(self):
        while not self.response_queue.empty():
            self.response_queue.get()

    def get_frequency(self):
        ret = self.send_and_recv("freq")
        if ret is None:
            return 0
        else:
            return float(ret)

    def set_frequency(self, value):
        self.log.info("Setting frequency to {}".format(value))
        ret = self.send_and_recv("move,set,{:.6f}".format(value))
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to set frequency, response was %s" % ret)
        ret = self.send_and_recv("move,go,{}".format(self.set_mode))
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to go to frequency, response was %s" % ret)

    def set_scanning_points(self, frequencies):
        self.log.info("Setting scan frequencies to {}".format(frequencies))
        ret = self.send_and_recv("peaks,replicates,1")
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to go to set the number of replicates, response was %s" % ret)
        ret = self.send_and_recv("peaks,width,{:f}".format(self.peak_width))
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to set the peak's width, response was %s" % ret)
        ret = self.send_and_recv("peaks,clear")
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to clear the peaks list, response was %s" % ret)
        for f in frequencies:
            ret = self.send_and_recv("peaks,add,{:.6f}".format(f))
            if ret is None or not ret.startswith("OK"):
                raise ScanningLaserReturnError("Unable to add a peak, response was %s" % ret)
        ret = self.send_and_recv("peaks,prepare")
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to prepare the scan, response was %s" % ret)

    def start_scan(self):
        ret = self.send_and_recv("peaks,start")
        if ret is None or not ret.startswith("OK"):
            raise ScanningLaserReturnError("Unable to start the scan, response was %s" % ret)

    def get_status(self):
        ret = self.send_and_recv("status")
        if ret is None:
            raise ScanningLaserReturnError("Error while probing status, no response.")
        try:
            d = json.loads(ret)
            return ScanningState.UNKNOWN(d)
        except:
            return ScanningState.UNKNOWN()
