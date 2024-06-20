from PySide2 import QtCore
import numpy as np
from qudi.core.module import LogicBase

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.util.datastorage import TextDataStorage
from qudi.util.datafitting import FitContainer, FitConfigurationsModel

class SpectrometerLogic(LogicBase):
    _scan_logic = Connector(name='scan_logic', interface='ScanningProbeLogic')

    _spectrum = StatusVar(name='spectrum', default=[None, None])
    _constant_acquisition = StatusVar(name='constant_acquisition', default=False)
    _fit_region = StatusVar(name='fit_region', default=[0, 1])
    _axis_type_frequency = StatusVar(name='axis_type_frequency', default=False)
    _fit_config = StatusVar(name='fit_config', default=dict())

    _sig_get_spectrum = QtCore.Signal(bool, bool, bool)
    sig_data_updated = QtCore.Signal()
    sig_state_updated = QtCore.Signal()
    sig_fit_updated = QtCore.Signal(str, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.refractive_index_air = 1.00028823
        self.speed_of_light = 2.99792458e8 / self.refractive_index_air
        self._fit_config_model = None
        self._fit_container = None

        self._logic_id = None

        # locking for thread safety
        self._lock = Mutex()

        self._spectrum = [None, None]
        self._wavelength = None
        self._stop_acquisition = False
        self._acquisition_running = False
        self._fit_results = None
        self._fit_method = ''

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._logic_id = self._scan_logic().module_uuid
        self._scan_logic().sigScanStateChanged.connect(self._update_scan_state)

        self._fit_config_model = FitConfigurationsModel(parent=self)
        self._fit_config_model.load_configs(self._fit_config)
        self._fit_container = FitContainer(parent=self, config_model=self._fit_config_model)
        self.fit_region = self._fit_region

        self._sig_get_spectrum.connect(self.get_spectrum, QtCore.Qt.QueuedConnection)
    
    def on_deactivate(self):
        """ Reverse steps of activation
        """
        self._scan_logic().sigScanStateChanged.disconnect(self._update_scan_state)
        self._sig_get_spectrum.disconnect()
        self._fit_config = self._fit_config_model.dump_configs()

    def stop(self):
        self._stop_acquisition = True

    def run_get_spectrum(self, constant_acquisition=None, differential_spectrum=None, reset=True):
        if constant_acquisition is not None:
            self.constant_acquisition = bool(constant_acquisition)
        if differential_spectrum is not None:
            self.log.warning("Differential spectra are not available in scqnning excitation spectroscopy.")
        self._sig_get_spectrum.emit(self._constant_acquisition, False, reset)

    def run_get_background(self, constant_acquisition=None, reset=True):
        self.log.warning("Background acquisition is not available for scanning excitation spectroscopy.")

    @property
    def acquisition_running(self):
        return self._acquisition_running

    @property
    def spectrum(self):
        if self._spectrum[0] is None:
            return None
        data = np.copy(self._spectrum[0])
        if self._repetitions_spectrum != 0:
            data /= self._repetitions_spectrum
        return data

    def get_spectrum_at_x(self, x):
        if self.x_data is None or self.spectrum is None:
            return -1
        if self.axis_type_frequency:
            return np.interp(x, self.x_data[::-1], self.spectrum[::-1])
        else:
            return np.interp(x, self.x_data, self.spectrum)

    @property
    def background(self):
        return np.zeros(np.shape(self.spectrum))

    @property
    def x_data(self):
        if self._axis_type_frequency:
            if self._wavelength is not None:
                return self.speed_of_light / self._wavelength
        else:
            return self._wavelength

    @property
    def repetitions(self):
        return self._repetitions_spectrum

    @property
    def background_correction(self):
        return False

    @background_correction.setter
    def background_correction(self, value):
        self.log.warning("Background correction is not available for scanning excitation spectroscopy.")
        self._background_correction = False
        self.sig_state_updated.emit()
        self.sig_data_updated.emit()

    @property
    def constant_acquisition(self):
        return self._constant_acquisition

    @constant_acquisition.setter
    def constant_acquisition(self, value):
        self._constant_acquisition = bool(value)
        self.sig_state_updated.emit()

    @property
    def differential_spectrum_available(self):
        return False

    @property
    def differential_spectrum(self):
        return False

    @differential_spectrum.setter
    def differential_spectrum(self, value):
        self.log.warning(f'differential_spectrum was requested, but it is not supported in scanning excitation spectroscopy.')
        self.sig_state_updated.emit()

    def save_spectrum_data(self, background=False, name_tag='', root_dir=None, parameter=None):
        """ Saves the current spectrum data to a file.

        @param bool background: Whether this is a background spectrum (dark field) or not.
        @param string name_tag: postfix name tag for saved filename.
        @param string root_dir: overwrite the file position in necessary
        @param dict parameter: additional parameters to add to the saved file
        """

        timestamp = datetime.now()

        # write experimental parameters
        parameters = {'acquisition repetitions': self.repetitions,
                      'differential_spectrum'  : self.differential_spectrum,
                      'background_correction'  : self.background_correction,
                      'constant_acquisition'   : self.constant_acquisition}
        # TODO: report the scanning parameters here.
        if self.fit_method != 'No Fit' and self.fit_results is not None:
            parameters['fit_method'] = self.fit_method
            parameters['fit_results'] = self.fit_results.params
            parameters['fit_region'] = self.fit_region
        if parameter:
            parameters.update(parameter)

        if self.x_data is None:
            self.log.error('No data to save.')
            return

        if self._axis_type_frequency:
            data = [self.x_data * 1e-12, ]
            header = ['Frequency (THz)', ]
        else:
            data = [self.x_data * 1e9, ]
            header = ['Wavelength (nm)', ]

        # prepare the data
        if not background:
            if self.spectrum is None:
                self.log.error('No spectrum to save.')
                return
            data.append(self.spectrum)
            file_label = 'spectrum' + name_tag
        else:
            self.log.warning("Trying to save the background, which is not available in scanning excitation spectroscopy.")
            if self.background is None or self.spectrum is None:
                self.log.error('No background to save.')
                return
            data.append(self.background)
            file_label = 'background' + name_tag

        header.append('Signal')

        if not background:
            # if background correction was on, also save the data without correction
            if self._background_correction:
                self._background_correction = False
                data.append(self.spectrum)
                self._background_correction = True
                header.append('Signal raw')

            # If the differential spectra arrays are not empty, save them as raw data
            if self._differential_spectrum and self._spectrum[1] is not None:
                data.append(self._spectrum[0])
                header.append('Signal ON')
                data.append(self._spectrum[1])
                header.append('Signal OFF')

        # save the date to file
        ds = TextDataStorage(root_dir=self.module_default_data_dir if root_dir is None else root_dir)

        file_path, _, _ = ds.save_data(np.array(data).T,
                                       column_headers=header,
                                       metadata=parameters,
                                       nametag=file_label,
                                       timestamp=timestamp,
                                       column_dtypes=[float] * len(header))

        # save the figure into a file
        figure, ax1 = plt.subplots()
        rescale_factor, prefix = self._get_si_scaling(np.max(data[1]))

        ax1.plot(data[0],
                 data[1] / rescale_factor,
                 linestyle=':',
                 linewidth=0.5
                 )

        if self.fit_method != 'No Fit' and self.fit_results is not None:
            if self._axis_type_frequency:
                x_data = self.fit_results.high_res_best_fit[0] * 1e-12
            else:
                x_data = self.fit_results.high_res_best_fit[0] * 1e9

            ax1.plot(x_data,
                     self.fit_results.high_res_best_fit[1] / rescale_factor,
                     linestyle=':',
                     linewidth=0.5
                     )

        ax1.set_xlabel(header[0])
        ax1.set_ylabel('Intensity ({} arb. u.)'.format(prefix))
        figure.tight_layout()

        ds.save_thumbnail(figure, file_path=file_path.rsplit('.', 1)[0])

        self.log.debug(f'Spectrum saved to:{file_path}')

    @staticmethod
    def _get_si_scaling(number):

        prefix = ['', 'k', 'M', 'G', 'T', 'P']
        prefix_index = 0
        rescale_factor = 1

        # Rescale spectrum data with SI prefix
        while number / rescale_factor > 1000:
            rescale_factor = rescale_factor * 1000
            prefix_index = prefix_index + 1

        intensity_prefix = prefix[prefix_index]
        return rescale_factor, intensity_prefix

    @property
    def axis_type_frequency(self):
        return self._axis_type_frequency

    @axis_type_frequency.setter
    def axis_type_frequency(self, value):
        self._axis_type_frequency = bool(value)
        self._fit_method = 'No Fit'
        self._fit_results = None
        self.fit_region = (0, 1e20)
        self.sig_data_updated.emit()

    @property
    def exposure_time(self): # TODO: act on the scanning_logic here.
        return self.spectrometer().exposure_time

    @exposure_time.setter
    def exposure_time(self, value): # TODO: act on the scanning_logic here.
        self.spectrometer().exposure_time = float(value)

    ################
    # Fitting things

    @property
    def fit_config_model(self):
        return self._fit_config_model

    @property
    def fit_container(self):
        return self._fit_container

    def do_fit(self, fit_method):
        if fit_method == 'No Fit':
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        self.fit_region = self._fit_region
        if self.x_data is None or self.spectrum is None:
            self.log.error('No data to fit.')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        if self._axis_type_frequency:
            start = len(self.x_data) - np.searchsorted(self.x_data[::-1], self._fit_region[1], 'left')
            end = len(self.x_data) - np.searchsorted(self.x_data[::-1], self._fit_region[0], 'right')
        else:
            start = np.searchsorted(self.x_data, self._fit_region[0], 'left')
            end = np.searchsorted(self.x_data, self._fit_region[1], 'right')

        if end - start < 2:
            self.log.error('Fit region limited the data to less than two points. Fit not possible.')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        x_data = self.x_data[start:end]
        y_data = self.spectrum[start:end]

        try:
            self._fit_method, self._fit_results = self._fit_container.fit_data(fit_method, x_data, y_data)
        except:
            self.log.exception(f'Data fitting failed:\n{traceback.format_exc()}')
            self.sig_fit_updated.emit('No Fit', None)
            return 'No Fit', None

        self.sig_fit_updated.emit(self._fit_method, self._fit_results)
        return self._fit_method, self._fit_results

    @property
    def fit_results(self):
        return self._fit_results

    @property
    def fit_method(self):
        return self._fit_method

    @property
    def fit_region(self):
        return self._fit_region

    @fit_region.setter
    def fit_region(self, fit_region):
        assert len(fit_region) == 2, f'fit_region has to be of length 2 but was {type(fit_region)}'

        if self.x_data is None:
            return
        fit_region = fit_region if fit_region[0] <= fit_region[1] else (fit_region[1], fit_region[0])
        new_region = (max(min(self.x_data), fit_region[0]), min(max(self.x_data), fit_region[1]))
        self._fit_region = new_region
        self.sig_state_updated.emit()
