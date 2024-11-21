from PySide2 import QtCore
import numpy as np
import matplotlib.pyplot as plt
import os
import uncertainties
from uncertainties import ufloat_fromstr, ufloat
from datetime import datetime

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage, get_timestamp_filename
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
    return np.exp(-x_prime**2/2 - y_prime**2/2) / (2*np.pi*np.sqrt(sigma_x*sigma_y))

class HyperspectralConfocalScannerLogic(LogicBase):
    _excitation_logic = Connector(name="excitation_logic", interface="ScanningExcitationLogic")
    _confocal_scanner = Connector(name="scanning_probe_logic", interface="ScanningProbeLogic")
    
    _frequencies = StatusVar(name="frequencies", default=list())
    _confocal_scans = StatusVar(name="confocal_scans", default=list())
    _wait_time_between_scans = StatusVar(name="wait_time_between_scans", default=10)
    _channel = StatusVar(name="channel", default="APD")
    _timestamp = StatusVar(name='timestamp', default=None)
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
        self._molecule_positions = []
    
    def on_activate(self):
        self._start_scan_timer.setSingleShot(True)
        self._start_scan_timer.timeout.connect(self.start_one_scan)
        self._confocal_scanner().sigScanStateChanged.connect(self.scan_state_updated)
        self._excitation_logic().sig_data_updated.connect(self.__spectrum_updated)
        self._stop_scan = False
    def on_deactivate(self):
        self._start_scan_timer.timeout.disconnect()
        self._confocal_scanner().sigScanStateChanged.disconnect(self.scan_state_updated)
        self._excitation_logic().sig_data_updated.disconnect(self.__spectrum_updated)
        
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
            self._confocal_scanner().stop_scan()
        
        
    def start_one_scan(self):
        self._confocal_scanner().start_scan(self.scan_axes, self.module_uuid)
        
    def _start_scan_with_delay(self):
        self.log.debug("start scan with a delay.")
        try:
            if self.thread() is not QtCore.QThread.currentThread():
                self._start_scan_timer.setInterval(self._wait_time_between_scans*1000)
                QtCore.QMetaObject.invokeMethod(self._start_scan_timer,
                                                "start",
                                                QtCore.Qt.BlockingQueuedConnection)
            else:    
                self._start_scan_timer.start(self._wait_time_between_scans*1000)
        except:
            self.log.exception("")
        
    def scan_state_updated(self, is_running, scan_data=None, caller_id=None):
        if caller_id != self.module_uuid:
            return
        with self._thread_lock:
            if scan_data is not None:
                if len(self._confocal_scans) <= self._current_scan_index:
                    self.log.debug("storing new scan")
                    self._confocal_scans.append(scan_data)
                else:
                    self.log.debug("updating current scan")
                    self._confocal_scans[self._current_scan_index] = scan_data
            if not is_running and not self._stop_scan:
                if self._current_scan_index < len(self._frequencies)-1:
                    self.log.debug("Triggering next scan")
                    self._current_scan_index += 1
                    self._excitation_logic().idle = self._frequencies[self._current_scan_index]
                    self._start_scan_with_delay()
                else:
                    self.log.debug("Fitting data")
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
                    if fit_result.errorbars:
                        x = fit_result.uvars['center_x']
                        y = fit_result.uvars['center_y']
                    else:
                        x = fit_result.best_values['center_x']
                        y = fit_result.best_values['center_y']
                    self._molecule_positions.append((x,y))         
                except:
                    self.log.exception('2D Gaussian fit unsuccessful.')
                    self._molecule_positions.append(None)  
            self.sig_molecule_positions_changed.emit()
            
    def __spectrum_updated(self):
        self.sig_spectrum_changed.emit()
            
    def add_frequency(self, v):
        with self._thread_lock:
            self._frequencies.append(v)
            self._frequencies.sort()
            self.sig_frequencies_changed.emit()
            
    def capture_current_frequency(self):
        self.add_frequency(self._excitation_logic().idle)
            
    def clear_frequencies(self):
        with self._thread_lock:
            self._frequencies = []
            self.sig_frequencies_changed.emit()
            
    @property
    def frequencies_of_interest(self):
        return self._frequencies
            
    @property
    def spectrum(self):
        return self._excitation_logic().spectrum
    @property
    def frequency(self):
        return self._excitation_logic().frequency
        
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
            if type(center_x) == uncertainties.core.Variable and type(center_y) == uncertainties.core.Variable:
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
        file_label = 'superresolution' + name_tag
        timestamp = datetime.now()
        ds = TextDataStorage(root_dir=self.module_default_data_dir if root_dir is None else root_dir,
                             include_global_metadata=True)
        # Construct the result matrix: one column for x, one for y, one for the probability map, and one for each scan
        x,y = self.probability_map_positions
        X,Y = np.meshgrid(x, y, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        data = np.zeros((len(X), 3 + len(self._confocal_scans)))
        data[:, 0] = X
        data[:, 1] = Y
        data[:, 2] = self.probability_map.ravel()
        # metadata
        metadata = {}
        # scan metadata, extract from the first scan
        units = {}
        scan_files = []
        for (i,scan_data) in enumerate(self._confocal_scans):
            parameters = {}
            for range, resolution, unit, axis in zip(scan_data.scan_range,
                                  scan_data.scan_resolution,
                                  scan_data.axes_units.values(),
                                  scan_data.scan_axes):

                units[axis] = unit
                parameters[f"{axis} axis name"] = axis
                parameters[f"{axis} axis unit"] = unit
                parameters[f"{axis} scan range"] = range
                parameters[f"{axis} axis resolution"] = resolution
                parameters[f"{axis} axis min"] = range[0]
                parameters[f"{axis} axis max"] = range[1]

            parameters["pixel frequency"] = scan_data.scan_frequency
            parameters[f"scanner target at start"] = scan_data.scanner_target_at_start
            parameters['measurement start'] = str(scan_data._timestamp)
            parameters['coordinate transform info'] = scan_data.coord_transform_info

            scan_file, _, _ = ds.save_data(scan_data.data[self._channel],
                                           metadata=parameters,
                                           nametag=file_label + f'_scan_{i}',
                                           timestamp=timestamp,
                                           column_headers='Image (columns is X, rows is Y)')
            scan_files.append(os.path.basename(scan_file))
        metadata['scan_files'] = scan_files
        scan_data = self._confocal_scans[0]
        # excitation spectrum metadata : just save the file and link to it.
        metadata['spectrum_path'] = os.path.basename(self._excitation_logic().save_spectrum_data(name_tag='superlocalisation'))
        # hyperspectral metadata
        metadata['frequency_points'] = self._frequencies
        metadata['wait_between_scans'] = self._wait_time_between_scans
        metadata['molecule_positions'] = self._molecule_positions
        metadata['channel'] = self._channel
        metadata['time_started'] = self._timestamp
        metadata['proba_map_resolution'] = self.proba_map_resolution
        # TODO: metadata, plot, saving
        # Saving
        unit_x = units[self.scan_axes[0]]
        unit_y = units[self.scan_axes[1]]
        header = [f"{self.scan_axes[0]} ({unit_x})", f"{self.scan_axes[1]} ({unit_y})", "PDF"]
        file_path, _, _ = ds.save_data(np.array(data).T,
                                       column_headers=header,
                                       metadata=metadata,
                                       nametag=file_label,
                                       timestamp=timestamp,
                                       column_dtypes=[float] * len(header))
        fig, ax = plt.subplots()
        cfimage = ax.imshow(self.probability_map.transpose(),
                            cmap='inferno',  # FIXME: reference the right place in qudi
                            origin='lower',
                            interpolation='none',
                            extent=(*np.asarray(scan_data.scan_range[0]),
                                    *np.asarray(scan_data.scan_range[1])))
        ax.set_aspect(1)
        ax.set_xlabel(self.scan_axes[0] + f' position ({scan_data.axes_units[self.scan_axes[0]]})')
        ax.set_ylabel(self.scan_axes[1] + f' position ({scan_data.axes_units[self.scan_axes[1]]})')
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        cbar = plt.colorbar(cfimage, shrink=0.8)
        cbar.ax.tick_params(which=u'both', length=0)
        ds.save_thumbnail(fig, file_path=file_path.rsplit('.', 1)[0])

        self.log.debug(f'superresolved map saved to:{file_path}')
        return file_path