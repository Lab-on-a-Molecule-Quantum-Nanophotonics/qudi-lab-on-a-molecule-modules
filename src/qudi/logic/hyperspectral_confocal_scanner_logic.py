from PySide2 import QtCore
import numpy as np
from uncertainties import ufloat_from_str

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.mutex import RecursiveMutex
from qudi.util.fit_models.gaussian import Gaussian2D, Gaussian

from qudi.interface.scanning_probe_interface import ScanData

def gaussian_proba_model(center_x, center_y, x, y):
    x0 = center_x.nominal_value
    y0 = center_y.nominal_value
    sigma_x = center_x.std_dev
    sigma_y = center_y.std_dev
    x_prime = (x - x0)/sigma_x
    y_prime = (y - y0)/sigma_y
    return np.exp(-x_prime**2/2 - y_prime**2/2) / (2*np.pi*np.sart(sigma_x*sigma_y))

class HyperspectralConfocalScannerLogic(LogicBase):
    _excitation_logic = Connector(name="excitation_logic", interface="ScanningExcitationLogic")
    _confocal_scanner = Connector(name="scanning_probe_logic", interface="ScanningProbeLogic")
    
    _frequencies = StatusVar(name="frequencies", default=list())
    _confocal_scans = StatusVar(name="confocal_scans", default=list())
    _excitation_spectrum = StatusVar(name="excitation_spectrum", default=None)
    _wait_time_between_scans = StatusVar(name="wait_time_between_scans", default=5)
    _molecule_positions = StatusVar(name="molecule_positions", default=list())
    _channel = StatusVar(name="channel", default="APD")
    proba_map_resolution = StatusVar(name="proba_map_resolution", default=500)
    
    scan_axes = ConfigOption(name="scan_axes", default=('x', 'y'))
    
    sig_molecule_positions_changed = QtCore.Signal()
    sig_spectrum_changed = QtCore.Signal()
    sig_frequencies_changed = QtCore.Signal()
    sig_scanning_changed = QtCore.Signal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = RecursiveMutex()
        self._current_scan_index = 0
        self._start_scan_timer = QtCore.QTimer(parent=self)
        self._stop_scan = False
    
    def on_activate(self):
        self._start_scan_timer.setSingleShot(True)
        self._start_scan_timer.timeout.connect(self.start_one_scan)
        self._confocal_scanner.sigScanStateChanged.connect(self.scan_state_updated)
        self._stop_scan = False
    def on_deactivate(self):
        self._start_scan_timer.timeout.disconnect()
        self._confocal_scanner.sigScanStateChanged.disconnect(self.scan_state_updated)
        
    @_confocal_scans.representer
    def __confocal_scans_to_dicts(self, scans):
        return [data.to_dict() for data in scans]
    @_confocal_scans.constructor
    def __confocal_scans_from_dicts(self, scan_dicts):
        try:
            scans = [ScanData.from_dict(scan_dict) for scan_dict in scan_dicts]
        except Exception as e:
            self.log.warning(f"Couldn't restore scans from StatusVar. Scans will be empty: {repr(e)}")
            scans = []

        return scans
        
    @_molecule_positions.representer
    def __molecule_positions_to_strs(self, positions):
        return [str for position in positions]
    @_molecule_positions.constructor
    def __molecule_positions_from_strs(self, positions_str):
        return [ufloat_from_str for position in positions_str]
        
    def start_scan(self):
        with self._thread_lock:
            if len(self._frequencies) <= 0:
                return
            self._current_scan_index = 0
            self._excitation_logic().idle = self._frequencies[0]
            self._confocal_scans = []
            self._stop_scan = False
            self.sig_scanning_changed.emit(True)
            self._start_scan_with_delay()
            
    def stop_scan(self):
        with self._thread_lock:
            self._stop_scan = True
            self._confocal_scanner.stop_scan()
        
        
    def start_one_scan(self):
        self._confocal_scanner.start_scan(self.scan_axes, self.module_uuid)
        
    def _start_scan_with_delay(self):
        try:
            if self.thread() is not QtCore.QThread.currentThread():
                QtCore.QMetaObject.invokeMethod(self,
                                                "_start_scan_with_delay",
                                                QtCore.Qt.BlockingQueuedConnection)
            else:    
                self._start_scan_timer.start(self._wait_time_between_scans)
        except:
            self.log.exception("")
        
    def scan_state_updated(self, is_running, scan_data=None, caller_id=None):
        if caller_id != self.module_uuid:
            return
        with self._thread_lock:
            if scan_data is not None:
                if len(self._confocal_scans) <= self._current_scan_index:
                    self._confocal_scans.append(scan_data)
                else:
                    self._confocal_scans[self._current_scan_index] = scan_data
            if not is_running and not self._stop_scan:
                if self._current_scan_index < len(self._frequencies)-1:
                    self._current_scan_index += 1
                    self._excitation_logic().idle = self._frequencies[self._current_scan_index]
                    self._start_scan_with_delay()
                else:
                    # all scans were run
                    self.calculate_superresolution()
                    
    def calculate_superresolution(self):
        model = Gaussian2D()
        with self._thread_lock:
            self._molecule_positions = []
            for scan in self._confocal_scans:
                try:
                    x = np.linspace(*scan.scan_range[0], scan.scan_resolution[0])
                    y = np.linspace(*scan.scan_range[1], scan.scan_resolution[1])
                    xy = np.meshgrid(x, y, indexing='ij')
                    data = scan.data[self._channel].ravel()
                    fit_result = model.fit(data, x=xy, **model.estimate_peak(data, xy))
                    x = fit_result.best_values['center_x']
                    y = fit_result.best_values['center_y']
                    self._molecule_positions.append((x,y))         
                except:
                    self.log.exception('2D Gaussian fit unsuccessful.')
                    self._molecule_positions.append(None)  
            self.sig_molecule_positions_changed.emit()
            
    def capture_spectrum(self):
        with self._thread_lock:
            spectrum = self._excitation_logic().spectrum
            frequency = self._excitation_logic().frequency
            self._excitation_spectrum = (frequency, spectrum)
            self.sig_spectrum_changed.emit()
            
    def add_frequency(self, v):
        with self._thread_lock:
            self._frequencies.append(v)
            self._frequencies.sort()
            self.sig_frequencies_changed.emit()
            
    def clear_frequencies(self):
        with self._thread_lock:
            self._frequencies = []
            self.sig_frequencies_changed.emit()
            
    @property
    def spectrum(self):
        return self._spectrum
        
    @property 
    def probability_map(self):
        if len(self._confocal_scans) <= 0:
            return np.zeros((0,0))
        scan = self._confocal_scans[0]
        x,y = self.probability_map_positions
        x,y = np.meshgrid(x, y, indexing='ij')
        x = x.ravel()
        y = y.ravel()
        matrix = np.zeros((self.proba_map_resolution, self.proba_map_resolution))
        N = len(self._molecule_positions)
        for (center_x, center_y) in self._molecule_positions:
            matrix += gaussian_proba_model(center_x, center_y, x, y).reshape(matrix.shape)
        matrix = matrix / N 
        return matrix
    
    @property
    def probability_map_positions(self):
        if len(self._confocal_scans) <= 0:
            return np.zeros(0), np.zeros(0)
        scan = self._confocal_scans[0]
        x = np.linspace(*scan.scan_range[0], self.proba_map_resolution)
        y = np.linspace(*scan.scan_range[1], self.proba_map_resolution)
        return x, y
        
    @property
    def molecule_positions(self):
        return self._molecule_positions
        
    def save(self, name_tag='', root_dir=None, parameter=None):
        # Construct the result matrix: one column for x, one for y, one for the signal
        # of each scan, and one for the probability map
        metadata = {}
        x,y = self.probability_map_positions
        x,y = np.meshgrid(x, y, indexing='ij')
        x = x.ravel()
        y = y.ravel()
        data = np.zeros((len(x), 3 + len(self._confocal_scans)))
        data[:, 0] = x 
        data[:, 1] = y 
        data[:, 2] = self.probability_map.ravel()
        for (i,scan) in enumerate(self._confocal_scans):
            data[:,i+3] = scan.data[self._channel].ravel()
        ds = TextDataStorage(root_dir=self.module_default_data_dir if root_dir is None else root_dir,
                             include_global_metadata=True)
        # TODO: metadata, plot, saving