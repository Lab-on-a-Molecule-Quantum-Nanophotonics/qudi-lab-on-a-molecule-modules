import time
import datetime

from PySide2 import QtCore
from fysom import Fysom
import numpy as np

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex


class CateyeLaserLogic(LogicBase):
    ldd_switches = Connector(interface="SwitchInterface")
    ldd_control = Connector(interface="ProcessControlInterface")
    cem_control = Connector(interface="ProcessControlInterface")
    cem_scan = Connector(interface="AutoScanInterface")
    fzw_control = Connector(interface="ProcessControlInterface")

    grating_channel = ConfigOption(name="grating_channel", default="grating")
    photodiode_channel = ConfigOption(name="photodiode_channel", default="photodiode")
    piezo_channel = ConfigOption(name="piezo_channel", default="piezo")
    frequency_channel = ConfigOption(name="frequency_channel", default="frequency")
    watchdog_delay = ConfigOption(name="watchdog_delay", default=0.2)
    scan_history_length = ConfigOption(name="scan_history_length", default=10)
    scan_data_history_length = ConfigOption(name="scan_data_history_length", default=10)

    _calibration_scan_span = StatusVar(name="calibration_scan_span", default=0.8)
    _calibration_scan_frequency = StatusVar(name="calibration_scan_frequency", default=5.0)
    _calibration_scan_duration = StatusVar(name="calibration_scan_duration", default=5.0)
    _calibration_encoder_start = StatusVar(name="calibration_encoder_start", default=4400)
    _calibration_encoder_stop = StatusVar(name="calibration_encoder_stop", default=16400)
    _calibration_encoder_step = StatusVar(name="calibration_encoder_step", default=10)
    _calibration_bias_start = StatusVar(name="calibration_encoder_start", default=0.0)
    _calibration_bias_stop = StatusVar(name="calibration_encoder_stop", default=50.0)
    _calibration_bias_step = StatusVar(name="calibration_encoder_step", default=5.0)
    _calibration_ramp_duty = StatusVar(name="calibration_ramp_duty", default=1.0)

    _mode_hop_threshold = StatusVar(name="mode_hop_threshold", default=0.1)
    
    _current_scan_index = StatusVar(name="current_scan_index", default=None)
    _scan_history = StatusVar(name="scan_history", default=[])
    _last_mode_scan = StatusVar(name="last_mode_scan", default=None)
    _mode_hops_indices = StatusVar(name="mode_hops", default=[])
    _current_scan_data_index = StatusVar(name="current_scan_data_index", default=None)
    _scan_data_history = StatusVar(name="scan_data_history", default=[])
    _current_calibration_index = StatusVar(name="current_calibration_index", default=None)
    _calibration_history = StatusVar(name="calibration_history", default=[])

    sigNewModeScanAvailable = QtCore.Signal()
    sigNewModeHopsAvailable = QtCore.Signal()
    sigPositionUpdated = QtCore.Signal(dict)
    sigScanningUpdated = QtCore.Signal(bool)
    sigCurrentScanUpdated = QtCore.Signal()
    sigProgressUpdated = QtCore.Signal(float)
    sigScanListUpdated = QtCore.Signal()
    sigCurrentDataUpdated = QtCore.Signal()
    sigCurrentCalibrationUpdated = QtCore.Signal()
    
    _sigWatchDog = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_scan = None
        self._report = {}
        self._last_report_time = time.perf_counter()
        self._grating_movement_start = time.perf_counter()
        self._step_start = time.perf_counter()
        self._grating_movement_retry = 0
        self._current_scan_data = None
        self._current_calibration = None
        self._data_lock = Mutex()
        self._watchdog_state = Fysom({
            "initial": "stopped",
            "events": [
                {"name":"start_idle", "src":"prepare_idle", "dst":"idle"},
                {"name":"start_idle", "src":"stopped", "dst":"prepare_idle"},

                {"name":"start_scan", "src":"idle", "dst":"prepare_step"},
                {"name":"position_grating", "src":"prepare_step", "dst":"grating_moving"},
                {"name":"start_scan_step", "src":"grating_moving", "dst":"record_scan_step"},
                {"name":"start_calibration_step", "src":"grating_moving", "dst":"record_calibration_step"},
                {"name":"start_center_frequency_measurement", "src":"record_calibration_step", "dst":"record_center_frequency"},
                {"name":"start_span_frequency_measurement", "src":"record_center_frequency", "dst":"record_span_frequency"},
                {"name":"step_done", "src":["record_scan_step", "record_span_frequency", "record_calibration_step"], "dst":"prepare_step"},
                {"name":"end_scan", "src":"prepare_step", "dst":"prepare_idle"},

                {"name":"interrupt_scan", "src":["prepare_step","grating_moving","record_scan_step","record_calibration_step"], "dst":"prepare_idle"},

                {"name":"stop_watchdog", "src":"*", "dst":"stopped"},
            ],
            "callbacks":{
                "on_start_scan": self._on_start_scan,
                "on_prepare_idle": self._on_stop_scan,
                "on_step_done": self._on_step_done,
            }
        })
        self._watchdog_lock = Mutex()
        self._single_step_scan_requested = False
        self._single_step_scan_started = False

    def on_activate(self):
        if len(self._scan_history) > 0:
            if self._current_scan_index is None:
                self._current_scan_index = len(self._scan_history) - 1
            self.set_current_scan(self._current_scan_index)
        else:
            self.create_new_scan()
        if len(self._calibration_history) > 0:
            if self._current_calibration_index is None:
                self._current_calibration_index = len(self._calibration_history) - 1
            self.set_current_calibration(self._current_calibration_index)
        if len(self._scan_data_history) > 0:
            if self._current_scan_data_index is None:
                self._current_scan_data_index = len(self._calibration_history) - 1
            self.set_current_scan_data(self._current_scan_data_index)
            
        self._sigWatchDog.connect(self._watchdog_callback, QtCore.Qt.QueuedConnection)
            
        self._report = {
            "units":{
                "frequency": self.fzw_control().constraints.channel_units[self.frequency_channel],
                "photodiode": self.cem_control().constraints.channel_units[self.photodiode_channel],
                "grating": self.cem_control().constraints.channel_units[self.grating_channel],
            },
            "values": {},
            "state": "",
        }
        self.start_watchdog()

    def on_deactivate(self):
        if self._current_scan is not None:
            self._scan_history[self._current_scan_index] = self._current_scan.to_dict()
        self.watchdog_event("stop_watchdog")
        while not self.watchdog_state == "stopped":
            time.sleep(0.1)
        self._sigWatchDog.disconnect()

    @property
    def watchdog_state(self):
        with self._watchdog_lock:
            return self._watchdog_state.current
    def watchdog_event(self, event):
        with self._watchdog_lock:
            self._watchdog_state.trigger(event)
            
    def start_watchdog(self):
        if self.watchdog_state != "stopped":
            return
        self.watchdog_event("start_idle")
        self._sigWatchDog.emit()

    def move_grating_to(self, value):
        constraints = self.cem_control().constraints
        if not constraints.channel_value_in_range("grating", value):
            raise ValueError(f"{value} grating value is out of allowed range.")
        self.cem_control().set_setpoint(self.grating_channel, value)
    def piezo_scan(self, frequency=None, span=None, offset=None, bias=None, mode_scan=True):
        self.ldd_switches().set_state("RAMP", "OFF")
        self.ldd_switches().set_state("HV,MOD", "RAMP")
        self.ldd_switches().set_state("CURRENT,MOD", "+RAMP")
        if frequency is None:
            self.ldd_control().set_setpoint("frequency", self._calibration_scan_frequency)
        else:
            self.ldd_control().set_setpoint("frequency", frequency)
        if span is None:
            self.ldd_control().set_setpoint("span", self._calibration_scan_span)
            span = self._calibration_scan_span
        else:
            self.ldd_control().set_setpoint("span", span)
        if offset is None:
            self.ldd_control().set_setpoint("offset", 0.0)
            offset = 0.0
        else:
            self.ldd_control().set_setpoint("offset", offset)
        if bias is None:
            self.ldd_control().set_setpoint("bias", 0.0)
        else:
            self.ldd_control().set_setpoint("bias", bias)
        self.ldd_control().set_setpoint("duty", self._calibration_ramp_duty)
        self.ldd_switches().set_state("RAMP", "ON")
        if mode_scan:
            return self.mode_scan()
        else:
            return None
    def mode_scan(self):
        self.cem_scan().trigger_scan()
        photodiode_scan = self.cem_scan().get_last_scan(self.photodiode_channel)
        piezo_scan = self.cem_scan().get_last_scan(self.piezo_channel)
        base = np.min(piezo_scan)
        norm_scan = piezo_scan - base
        maxi = np.max(norm_scan)
        norm_scan = norm_scan / maxi
        norm_scan = norm_scan * self._current_scan.span
        norm_scan = norm_scan + self._current_scan.offset
        sortperm = np.argsort(norm_scan)
        self._last_mode_scan = np.vstack((norm_scan[sortperm], photodiode_scan[sortperm]))
        self.sigNewModeScanAvailable.emit()
        return self._last_mode_scan

    def find_mode_hops(self):
        diff_array = np.diff(self._last_mode_scan[1,:], prepend=np.nan)
        self.log.debug(f"diff_array = {diff_array.shape}")
        jump_indices = np.nonzero(np.abs(diff_array) > self._mode_hop_threshold)[0]
        self.log.debug(f"jump_indices = {jump_indices}")
        self.log.debug(f"concatenate {(np.array([0]), jump_indices, np.array([self._last_mode_scan.shape[-1]-1]))}")
        self._mode_hops_indices = np.concatenate((np.array([0]), jump_indices, np.array([self._last_mode_scan.shape[-1]-1])))
        self.sigNewModeHopsAvailable.emit()
        return self.mode_hops

    def find_best_scan_offset_span(self):
        hop_tensions = self._last_mode_scan[0,self._mode_hops_indices]
        ranges = np.diff(hop_tensions)
        i_max = np.argmax(ranges)
        offset = (hop_tensions[i_max] + hop_tensions[i_max])/2
        span = ranges[i_max]
        return offset, span

    @property
    def last_mode_scan(self):
        return self._last_mode_scan

    @property 
    def mode_hops(self):
        return self._last_mode_scan[0,self._mode_hops_indices]

    @property
    def is_scanning(self):
        return self.watchdog_state in ["prepare_scan_step", "grating_moving", "record_scan_step"]

    def read_status(self):
        frequency = self.fzw_control().get_process_value(self.frequency_channel)
        photodiode = self.cem_control().get_process_value(self.photodiode_channel)
        grating = self.cem_control().get_process_value(self.grating_channel)
        self._last_report_time = time.perf_counter()
        self._report["values"].update({
            "frequency":frequency,
            "photodiode":photodiode,
            "grating":grating,
        })
        self._report["state"] = self.watchdog_state
        return self._report

    @property
    def new_report_needed(self):
        now = time.perf_counter() 
        if self.watchdog_state in ("record_center_frequency",):
            return True
        elif self.watchdog_state in ("record_scan_step", "record_span_frequency"):
            return self._current_scan.sample_time <= (now-self._last_report_time)
        else:
            return self.watchdog_delay <= (now-self._last_report_time)

    def _on_start_scan(self, *args, **kwargs):
        self.sigScanningUpdated.emit(True)

    def _on_stop_scan(self, *args, **kwargs):
        self.sigScanningUpdated.emit(False)

    def _on_step_done(self, *args, **kwargs):
        if not self._single_step_scan_started:
            self._current_scan.next()

    def _watchdog_callback(self):
        try:
            start = time.perf_counter()
            watchdog_state = self.watchdog_state
            # First, report the current position of the laser
            new_status_read = self.new_report_needed
            if new_status_read:
                self.sigPositionUpdated.emit(self.read_status())
            if watchdog_state == "prepare_idle":
                self.ldd_switches().set_state("RAMP", "OFF")
                self.watchdog_event("start_idle")
            elif watchdog_state == "idle":
                pass
            elif watchdog_state == "prepare_step":
                if self._single_step_scan_started:
                    self.log.info("Single step scan finished.")
                    self._single_step_scan_started = False
                    self.watchdog_event("end_scan")
                elif self._current_scan is None:
                    self.log.error("Cannot scan when no scan has been prepared!")
                    self.watchdog_event("interrupt_scan")
                elif self._current_scan.finished():
                    self.log.info("Scan finished.")
                    self.watchdog_event("end_scan")
                else:
                    if self._current_scan.is_first_step() or self._single_step_scan_requested:
                        if self._current_scan.calibration:
                            with self._data_lock:
                                self._current_calibration = Calibration.from_scan_configuration(self._current_scan.to_dict())                            
                        else:
                            with self._data_lock:
                                self._current_scan_data = ScanData(self._current_scan.to_dict())
                    if self._single_step_scan_requested:
                        self.log.debug("single step scan.")
                        self._single_step_scan_requested = False
                        self._single_step_scan_started = True
                    config = self._current_scan.current_step()
                    self.ldd_switches().set_state("RAMP", "OFF")
                    self.ldd_switches().set_state("HV,MOD", "RAMP")
                    self.ldd_switches().set_state("CURRENT,MOD", "+RAMP")
                    for channel in ("frequency", "span", "offset", "bias"):
                        self.ldd_control().set_setpoint(channel, config[channel])
                    self.ldd_control().set_setpoint("duty", self._calibration_ramp_duty)
                    self.cem_control().set_setpoint("scan_duration", 5.0/config["frequency"])
                    self.cem_control().set_setpoint(self.grating_channel, config["grating"])
                    if self._current_scan.calibration:
                        while self._current_calibration.grating[self._current_calibration.step_no] < self._current_scan.grating:
                            self._current_calibration.step_no += 1
                    self.log.debug("Scan step prepared")
                    self._grating_movement_start = time.perf_counter()
                    self._grating_movement_retry = 0
                    self.watchdog_event("position_grating")
            elif watchdog_state == "grating_moving":
                setpoint_reached = np.abs(self._current_scan.grating - self.cem_control().get_setpoint(self.grating_channel)) < 1
                timeouted = (start - self._grating_movement_start) > 10.0
                if timeouted and not setpoint_reached:
                    self.log.warning(f"The motor is stuck, I will try to reset by homing it ({self._grating_movement_retry}/3).")
                    if self._grating_movement_retry >= 3:
                        self.log.error("Could not un-stuck the motor after three try. Aborting the scan.")
                        self.watchdog_event("interrupt_scan")
                    else:
                        self._grating_movement_retry += 1
                        self.cem_control()._reset_motor() # TODO, this is not defined in any interface!
                        self._grating_movement_start = time.perf_counter()
                elif setpoint_reached:
                    self.log.debug("Grating position reached, starting the ramp.")
                    self.ldd_switches().set_state("RAMP", "ON")
                    if self._current_scan.calibration:
                        self.watchdog_event("start_calibration_step")
                    else:
                        self._step_start = time.perf_counter()
                        self.watchdog_event("start_scan_step")
            elif watchdog_state == "record_scan_step":
                if new_status_read:
                    self._current_scan_data.append(
                        frequency=self._report['values']["frequency"],
                        photodiode=self._report['values']["photodiode"],
                        step=self._current_scan.step_no,
                        repeat=self._current_scan.repeat_no,
                    )
                    self.sigCurrentDataUpdated.emit()
                step_finished = start - self._step_start > 1/self._current_scan.frequency
                if step_finished:
                    self.watchdog_event("step_done")
            elif watchdog_state == "record_calibration_step":
                # TODO scan the frequencies and look for mode-hops there.
                self.log.debug("recording calibration step")
                self.mode_scan()
                self.find_mode_hops()
                offset, span = self.find_best_scan_offset_span()
                # TODO: this is imperfect, a wide range of piezo scan does not mean we have a wide range of wavelengths available.
                if span > self._current_calibration.span[self._current_calibration.step_no]: 
                    self._current_calibration.grating[self._current_calibration.step_no] = self._current_scan.grating
                    self._current_calibration.bias[self._current_calibration.step_no] = self._current_scan.bias
                    self._current_calibration.offset[self._current_calibration.step_no] = offset
                    self._current_calibration.span[self._current_calibration.step_no] = span
                    self.ldd_control().set_setpoint("ramp_halt", offset)
                    self.ldd_switches().set_state("RAMP", "OFF")
                    self.watchdog_event("start_center_frequency_measurement")
                else:
                    self.watchdog_event("step_done")
            elif watchdog_state == "record_center_frequency":
                self.log.debug("reading center frequency")                
                freq = self._report['values']["frequency"]
                self._current_calibration.center_frequencies[self._current_calibration.step_no] = freq
                self.ldd_switches().set_state("RAMP", "ON")
                self._step_start = time.perf_counter()
                self.watchdog_event("start_span_frequency_measurement")
            elif watchdog_state == "record_span_frequency":
                if new_status_read:
                    self.log.debug("reading span frequency")
                    freq = self._report['values']["frequency"]
                    current_mini = self._current_calibration.mini_frequencies[self._current_calibration.step_no]
                    if current_mini == 0:
                        self._current_calibration.mini_frequencies[self._current_calibration.step_no] = freq 
                    else:
                        self._current_calibration.mini_frequencies[self._current_calibration.step_no] = min(current_mini, freq)
                    current_maxi = self._current_calibration.maxi_frequencies[self._current_calibration.step_no]
                    if current_maxi == 0:
                        self._current_calibration.maxi_frequencies[self._current_calibration.step_no] = freq 
                    else:
                        self._current_calibration.maxi_frequencies[self._current_calibration.step_no] = max(current_maxi, freq)
                step_finished = start - self._step_start > 1/self._current_scan.frequency
                if step_finished:
                    self.sigCurrentCalibrationUpdated.emit()
                    self.watchdog_event("step_done")
            elif watchdog_state == "stopped":
                return
            else: 
                raise ValueError(f"Watchdog in unhandled state {watchdog_state}")
            self._sigWatchDog.emit()
        except:
            self.log.exception("")

    @property
    def current_scan(self):
        return self._current_scan
    @property
    def current_scan_index(self):
        return self._current_scan_index
    def delete_current_scan(self):
        if len(self._scan_history) <= 0:
            self.log.warning("Trying to empty an empty scan history.")
            return
        self._scan_history.pop(self._current_scan_index)
        self._current_scan_index = max(0, self._current_scan_index-1)
        self._current_scan = ScanConfiguration.from_dict(self._scan_history[self._current_scan_index])
        self.sigScanListUpdated.emit()
        self.sigCurrentScanUpdated.emit()
    def create_new_scan(self):
        self.log.debug("Create new scan.")
        if self._current_scan is not None:
            self._scan_history[self._current_scan_index] = self._current_scan.to_dict()
        self._current_scan = ScanConfiguration()
        self._current_scan.create_zero_step()
        self._scan_history.append(self._current_scan.to_dict())
        self._current_scan_index = len(self._scan_history)-1
        self.sigScanListUpdated.emit()
        self.sigCurrentScanUpdated.emit()
    def start_scan(self):
        self.watchdog_event("start_scan")
    def set_current_scan(self, i):
        self.log.debug("set current scan")
        # Save the current scan
        if self._current_scan is not None:
            self._scan_history[self._current_scan_index] = self._current_scan.to_dict()
        self._current_scan_index = min(max(i, 0), len(self._scan_history)-1)
        self._current_scan = ScanConfiguration.from_dict(self._scan_history[self._current_scan_index])
        self.sigCurrentScanUpdated.emit()
    def set_current_scan_name(self, name):
        self._current_scan.name = name
        self._scan_history[self._current_scan_index] = self._current_scan.to_dict()
        self.sigScanListUpdated.emit()
    def set_current_scan_calibration(self, calibration):
        self._current_scan.calibration = calibration
    def insert_new_step(self, i):
        if len(self._current_scan) <= 0:
            self._current_scan.create_zero_step()
        else:
            self._current_scan.duplicate_step(i)
        self.sigCurrentScanUpdated.emit()
    def delete_step(self, i):
        self._currentscan.delete_step(i)
        self.sigCurrentScanUpdated.emit()
    def move_step_up(self, i):
        if i>0:
            self._current_scan.swap(i, i-1)
        self.sigCurrentScanUpdated.emit()
    def move_step_down(self, i):
        if i<len(self._current_scan)-1:
            self._current_scan.swap(i, i+1)
        self.sigCurrentScanUpdated.emit()
    def set_step_grating(self, i, value):
        self._current_scan.set_grating(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_span(self, i, value):
        self._current_scan.set_span(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_offset(self, i, value):
        self._current_scan.set_offset(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_frequency(self, i, value):
        self._current_scan.set_frequency(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_bias(self, i, value):
        self._current_scan.set_bias(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_sample_time(self, i, value):
        self._current_scan.set_sample_time(i, value)
        self.sigCurrentScanUpdated.emit()
    def set_step_repeat(self, i, value):
        self._current_scan.set_repeat(i, value)
        self.sigCurrentScanUpdated.emit()
    def scan_step(self, step_no):
        self._single_step_scan_requested = True
        self.watchdog_event("start_scan")
    def set_position_to(self, step_no, val):
        pass
    @property 
    def scan_history_names(self):
        return [f"{scan['date_created']}: {scan['name']}" for scan in self._scan_history]
        

# TODO: handle scanning in the background. The phases of scanning are:
# 1. set grating
# 2. wait for grating to be in position
# 3. start the scan
# 4. while the scan is running, sample the wavelength, pd, (and later the APD)
# 5. get the next configuration and loop back if needed.
# See when signaling that new data are available makes sense.
# TODO: calibration. Scan the available grating positions, and for each position
# find the widest scan range. The scanning algorithm is the same as the normal
# scan.
# TODO: build a scan from a range of wavelengths. Tile the spanusing the widest
# scan ranges available
# TODO: find a frequency from the calibration with a frequency error tolerance.
# TODO: scan history

class ScanConfiguration:
    count = 0
    def __init__(self, grating=np.empty(0,dtype=int), span=np.empty(0,dtype=float), offset=np.empty(0,dtype=float), frequency=np.empty(0,dtype=float), bias=np.empty(0,dtype=float), sample_time=np.empty(0,dtype=float), repeat=np.empty(0,dtype=int), calibration=False, date_created=None, name=None):
        assert len(grating) == len(span) == len(offset) == len(frequency) == len(bias) == len(sample_time) == len(repeat)
        if name is None:
            self.name = f"Scan {ScanConfiguration.count}"
            ScanConfiguration.count += 1
        else:
            self.name = name
        self._grating = grating
        self._span = span
        self._offset = offset
        self._frequency = frequency
        self._bias = bias
        self._sample_time = sample_time
        self._repeat = repeat
        self.calibration = calibration
        self.i = 0
        self.j = 0
        if date_created is None:
            self.date_created = datetime.datetime.now()
        else:
            self.date_created = date_created
    def swap_steps(self, i, j):
        self._grating[i], self._grating[j] = self._grating[j], self._grating[i]
        self._span[i], self._span[j] = self._span[j], self._span[i]
        self._offset[i], self._offset[j] = self._offset[j], self._offset[i]
        self._frequency[i], self._frequency[j] = self._frequency[j], self._frequency[i]
        self._bias[i], self._bias[j] = self._bias[j], self._bias[i]
        self._sample_time[i], self._sample_time[j] = self._sample_time[j], self._sample_time[i]
        self._repeat[i], self._repeat[j] = self._repeat[j], self._repeat[i]
    def start_scan(self):
        self.i = 0
        self.j = 0
    def step(self, i):
        return {"grating":     self._grating[i], 
                "span":        self._span[i], 
                "offset":      self._offset[i], 
                "frequency":   self._frequency[i], 
                "bias":        self._bias[i], 
                "sample_time": self._sample_time[i],
                "calibration": self.calibration,
                "repeat":      self._repeat[i],
                "step_no":     i,
                } 
    def current_step(self):
        return {"grating":     self.grating, 
                "span":        self.span, 
                "offset":      self.offset, 
                "frequency":   self.frequency, 
                "bias":        self.bias, 
                "sample_time": self.sample_time,
                "calibration": self.calibration,
                "repeat_no":   self.j,
                "step_no":     self.i,
                } 
    def next(self):
        self.j += 1
        if self.j >= self.repeat:
            self.j = 0
            self.i += 1
    def finished(self):
        return self.i >= len(self._grating)
    def is_first_step(self):
        return self.i == 0 and self.j == 0
    def to_dict(self):
        return {"grating":      self._grating, 
                "span":         self._span, 
                "offset":       self._offset, 
                "frequency":    self._frequency, 
                "bias":         self._bias, 
                "sample_time":  self._sample_time,
                "repeat":       self._repeat,
                "calibration":  self.calibration,
                "date_created": self.date_created,
                "name":         self.name,
                } 
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    def __repr__(self):
        return "ScanConfiguration(" + ",".join([f"{k}={v}" for (k,v) in self.to_dict().items()]) + ")"
    @property
    def grating(self):
        return self._grating[self.i]
    @property 
    def span(self):
        return self._span[self.i]
    @property 
    def offset(self):
        return self._offset[self.i]
    @property
    def frequency(self):
        return self._frequency[self.i]
    @property
    def bias(self):
        return self._bias[self.i]
    @property
    def sample_time(self):
        return self._sample_time[self.i]
    @property
    def repeat(self):
        return self._repeat[self.i]
    def set_grating(self, i, v):
        self._grating[i] = v
    def set_span(self, i, v):
        self._span[i] = v
    def set_offset(self, i, v):
        self._offset[i] = v
    def set_frequency(self, i, v):
        self._frequency[i] = v
    def set_bias(self, i, v):
        self._bias[i] = v
    def set_sample_time(self, i, v):
        self._sample_time[i] = v
    def set_repeat(self, i, v):
        self._repeat[i] = v
    def create_zero_step(self):
        self._grating = np.array([4400])
        self._span = np.array([1.0])
        self._offset = np.array([0.0])
        self._frequency = np.array([5.0])
        self._bias = np.array([0.0])
        self._sample_time = np.array([0.01])
        self._repeat = np.array([1])
    def duplicate_step(self, i):
        self._grating = np.insert(self._grating, i, self._grating[i])
        self._span = np.insert(self._span, i, self._span[i])
        self._offset = np.insert(self._offset, i, self._offset[i])
        self._frequency = np.insert(self._frequency, i, self._frequency[i])
        self._bias = np.insert(self._bias, i, self._bias[i])
        self._sample_time = np.insert(self._sample_time, i, self._sample_time[i])
        self._repeat = np.insert(self._repeat, i, self._repeat[i])
    def delete_step(self, i):
        self._grating = np.delete(self._grating, i)
        self._span = np.delete(self._span, i)
        self._offset = np.delete(self._offset, i)
        self._frequency = np.delete(self._frequency, i)
        self._bias = np.delete(self._bias, i)
        self._sample_time = np.delete(self._sample_time, i)
        self._repeat = np.delete(self._repeat, i)
    @property
    def repeat_no(self):
        return self.j
    @property
    def step_no(self):
        return self.i
    def __len__(self):
        return len(self._grating)
    def __getitem__(self, i):
        return self.step(i)
   
class ScanData:
    frequency_col = 0
    photodiode_col = 1
    step_col = 2
    repeat_col = 3
    def __init__(self, configuration, date_started=datetime.datetime.now()):
        self.configuration = configuration
        self._data = np.empty((0, 4), dtype=float)
        self.date_started = date_started
    def append(self, frequency, photodiode, step, repeat):
        np.append(self._data, [[frequency, photodiode, step, repeat]])
    @property
    def data(self):
        return self._data
    @property
    def frequency(self):
        return self._data[:, self._frequency_col]
    @property
    def photodiode(self):
        return self._data[:, self._photodiode_col]
    @property
    def step(self):
        return self._data[:, self._step_col]
    @property
    def repeat(self):
        return self._data[:, self._repeat_col]
    def to_dict(self):
        return {"data":     self._data, 
                "configuration": self.configuration,
                "date_started": self.date_started,
                } 
    @classmethod
    def from_dict(cls, d):
        v = cls(d["configuration"], d["date_started"])
        v._data = d["data"]
        return v

class Calibration:
    count = 0
    def __init__(self, grating=None, bias=None, offset=None, span=None, center_frequencies=None, mini_frequencies=None, maxi_frequencies=None, name=None, date_created=None):
        if name is None:
            self.name = f"Calibration {Calibration.count}"
            Calibration.count += 1
        else:
            self.name = name
        self.grating = grating if grating is not None else np.empty(0, int)
        self.bias = bias if bias is not None else np.empty(0, float)
        self.offset = offset if offset is not None else np.empty(0, float)
        self.span = span if span is not None else np.empty(0, float)
        self.center_frequencies = center_frequencies if center_frequencies is not None else np.empty(0, float)
        self.mini_frequencies = mini_frequencies if mini_frequencies is not None else np.empty(0, float)
        self.maxi_frequencies = maxi_frequencies if maxi_frequencies is not None else np.empty(0, float)
        assert len(self.grating) == len(self.bias) == len(self.offset) == len(self.span) == len(self.center_frequencies) == len(self.mini_frequencies) == len(self.maxi_frequencies)
        if date_created is None:
            self.date_created = datetime.datetime.now()
        else:
            self.date_created = date_created
        self.step_no = 0
    def to_dict(self):
        return {"grating": self.grating, 
                "bias": self.bias,
                "offset": self.offset,
                "span": self.span,
                "center_frequencies": self.center_frequencies,
                "mini_frequencies": self.mini_frequencies,
                "maxi_frequencies": self.maxi_frequencies,
                "name": self.name,
                "date_created": self.date_created,
                } 
    @classmethod
    def from_dict(cls, d):
        v = cls(**d)
        return v
    @classmethod
    def from_scan_configuration(cls, d):
        grating = np.unique(d["grating"])
        return cls(
                    grating = grating,
                    bias = np.zeros(len(grating)),
                    offset = np.zeros(len(grating)),
                    span = np.zeros(len(grating)),
                    center_frequencies = np.zeros(len(grating)),
                    mini_frequencies = np.zeros(len(grating)),
                    maxi_frequencies = np.zeros(len(grating))
                    )
            
    