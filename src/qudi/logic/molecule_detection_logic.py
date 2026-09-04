import datetime
import queue
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from queue import Empty, Full

import matplotlib.pyplot as plt
import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from qudi.core.configoption import ConfigOption
from qudi.core.module import LogicBase
from qudi.core.statusvariable import StatusVar
from qudi.util.fit_models.lorentzian import Lorentzian
from qudi.util.units import ScaledFloat
from scipy.signal import convolve
from scipy.special import digamma, polygamma
from scipy.stats import gamma, norm
from tifffile import TiffFile, imread

from qudi.util.fit_models.gaussian import Gaussian2D
from qudi.util.tomldatastorage import TOMLHeaderDataStorage


def trigamma(z):
    return polygamma(1, z)


def inv_gauss(x):
    return norm.isf(x, loc=0, scale=1)


class LocalMaximumDecision(Enum):
    Discard = -1
    Undecided = 0
    Keep = 1

@dataclass
class WorkerMessage:
    data: np.ndarray | None = None
    slice_number: int = 0

@dataclass
class BuilderMessage:
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    slice_number: int = 0

class WorkerSignals(QObject):
    success = Signal(int)  # thread_id
    errored = Signal(tuple)
    finished = Signal(int)  # thread_id
    def __init__(self, worker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker = worker

    @Slot()
    def interrupt(self):
        self.worker._interrupt = True

class Worker(QRunnable):
    """
    A worker thread.
    Parameters:
    * `fn`: The function which is to be run.
    * `args`: arguments to feed the function.
    * `kwargs`: keyword arguments to feed the function.
    """

    def __init__(self, initialization: Callable, loop: Callable, finish: Callable=lambda:None, *args, thread_id=0, **kwargs):
        super().__init__()
        self.initialization_callback = initialization
        self.loop_callback = loop
        self.finish_callback = finish
        self.thread_id = thread_id
        self.signals = WorkerSignals(self)
        self.variables = {}
        self.args = args
        self.kwargs = kwargs
        self._interrupt = False

    @Slot()
    def run(self):
        try:
            self.variables = self.initialization_callback(*self.args, **self.kwargs)
            loop_running = True
            while loop_running:
                keep_running,self.variables = self.loop_callback(**self.variables)
                loop_running = not self._interrupt and keep_running
            if not self._interrupt:
                self.finish_callback(**self.variables)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.errored.emit((exctype, value, traceback.format_exc())) 
        else:
            self.signals.success.emit(self.thread_id)
        finally:
            self.signals.finished.emit(self.thread_id)


ANDOR_METADATA_KEYS = ("Temperature", "ExposureTime", "KineticCycleTime", "Accumulations", "PhotonCounting", "EmDacLevel", "Frames", "VerticalShiftSpeed", "PreAmpSetting", "CameraSerial", "BaselineClamp", "Model", "ChipSizeX", "ChipSizeY", "SoftwareVersion")

@dataclass
class UserActionableMoleculeDetectionSettings:
    """
    Settings of the molecule detection algorithm actionable by the user.
    """

    input_file: str | None = None
    quantile_gamma_fit: float | None = None
    quantile_max_estimate: float | None = None
    lorentzian_safety_margin: float | None = None
    probability_false_positive: float | None = None
    frame_per_slice: float | None = None
    number_of_frames_sampled_for_gamma_fitting: int | None = None
    number_of_frames_sampled_for_psf_estimation: int | None = None
    border_excluded_pixels: int | None = None


@dataclass
class MoleculeDetectionSettings:
    """
    Parameters of the molecule detection algorithm, including ones not actionable
    by the user.
    """

    input_file: str
    quantile_gamma_fit: float
    quantile_max_estimate: float
    lorentzian_safety_margin: float
    probability_false_positive: float
    frame_per_slice: float
    frequency_fwhm: float
    position_stdev: float
    number_of_workers: float
    theta: float
    assumed_amplitude: float
    kernel_spatial_length: float
    kernel_frequency_length: float
    number_of_frames_sampled_for_gamma_fitting: int 
    number_of_frames_sampled_for_psf_estimation: int
    border_excluded_pixels: int 

@dataclass
class MoleculePositionsResult:
    """
    A dataclass to hold the result of the molecule detection algorithm.
    """

    file_name: str
    file_metadata: dict[str, float]
    algorithm_settings: MoleculeDetectionSettings
    coordinates_array: np.ndarray


@dataclass
class SingleMoleculeLocalizationDataset:
    """
    Data related to the localization of a single molecule.
    """
    input_file: str
    position_rough_estimate_l: int
    position_rough_estimate_m: int
    position_rough_estimate_n: int
    flattened_image: np.ndarray
    flattened_image_m: np.ndarray
    flattened_image_n: np.ndarray
    frequency_spectrum: np.ndarray
    fit_result_spatially: dict | None
    fit_result_spectrally: dict | None
    

def positivity_constraint_constructor(val):
    if val <= 0:
        return 1.0
    return val

class MoleculeDetectionLogic(LogicBase):
    """
    ```yaml
    molecule_detection_logic:
        module.Class: 'molecule_detection_logic.MoleculeDetectionLogic'
        options:
            number_of_workers: 4 # default
    ```
    """

    _number_of_workers = ConfigOption(name="number_of_workers", default=4)

    _quantile_gamma_fit = StatusVar(name="quantile_gamma_fit", default=0.99)
    _number_of_frames_sampled_for_gamma_fitting = StatusVar(default=10)
    _number_of_frames_sampled_for_psf_estimation = StatusVar(default=50)
    _quantile_max_estimate = StatusVar(name="quantile_max_estimate", default=0.9999)
    _lorentzian_safety_margin = StatusVar(name="lorentzian_safety_margin", default=3)

    _frequency_fwhm = StatusVar(name="frequency_fwhm", default=2.7)
    _position_stdev = StatusVar(name="pixel_stdev", default=(1 / 2.3548) * 2.7)
    _probability_false_positive = StatusVar(
        name="probability_false_positive", default=1e-6
    )
    _theta = StatusVar(name="theta", default=1.0)
    _assumed_amplitude = StatusVar(name="assumed_amplitude", default=1000)
    _kernel_spatial_length = StatusVar(default=5)
    _kernel_frequency_length = StatusVar(default=3)
    _input_file = StatusVar(default=None)
    _frames_per_slice = StatusVar(default=100)
    _excluded_pixels_border = StatusVar(default=1)

    progress_update = Signal(float)  # Localization progress between 0 and 1
    failure = Signal()
    finished = Signal()
    new_result = Signal(MoleculePositionsResult)
    settings_updated = Signal(MoleculeDetectionSettings)
    position_updated = Signal(int)
    new_superlocalization = Signal(SingleMoleculeLocalizationDataset)
    _start_watchdog_timer = Signal()
    _start_workers = Signal()
    _start_builder = Signal()
    _localize = Signal()
    _trigger_interrupt = Signal()
    gamma_estimate_histogram_updated = Signal(np.ndarray, np.ndarray)
    gamma_law_fit_updated = Signal(np.ndarray, np.ndarray)

    @_theta.constructor
    def theta_constructor(self, yaml_data):
        val = float(yaml_data)
        if val <= 0:
            self.log.warning("Stored theta value is <= 0. Defaulting to 1.0.")
            return 1.0
        return val

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slicer_to_worker_queue = queue.Queue()
        self._worker_to_builder_queue = queue.Queue()
        self._background_kernel = np.array([])
        self._threadpool = QThreadPool(self)
        self._workers = []
        self._slicer = None
        self._builder = None
        self._total_number_of_slices = 0
        self._number_of_frames = 0
        self._interrupt = False
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = []
        self._slicer_started = False
        self._builder_started = False
        self._workers_started = False
        self._timer = QTimer(parent=self)
        self._noise_fitting_histogram = (np.array([]), np.array([]))
        self._fitted_gamma_law = (np.array([]), np.array([]))
        self._flattened_image = np.array([])
        self._metadata = None
        self._last_result = None
        self._last_superlocalization = None
        self._input_file_width = 0
        self._input_file_height = 0

    def on_activate(self) -> None:
        self._start_watchdog_timer.connect(self.__start_watchdog_timer, QtCore.Qt.QueuedConnection)
        self._start_workers.connect(self.__start_workers, QtCore.Qt.QueuedConnection)
        self._start_builder.connect(self.__start_builder, QtCore.Qt.QueuedConnection)
        self._trigger_interrupt.connect(self.__interrupt, QtCore.Qt.QueuedConnection)
        self._localize.connect(self.__localize, QtCore.Qt.QueuedConnection)

    def on_deactivate(self) -> None:
        self._start_watchdog_timer.disconnect()
        self._start_workers.disconnect()
        self._start_builder.disconnect()
        self._localize.disconnect()
        self.interrupt()
        if self.module_state() == 'locked':
            self.module_state.unlock()

    def ln_transform(self, r_lmn: np.ndarray) -> np.ndarray:
        """
        log transform function use for the gamma distribution background
        Parameters :
        * `r_lmn` : array given to get its log transform
        """
        r_lmn[np.where(r_lmn == 0)] = 1  # disaster prevention
        theta = float(self._theta)
        return np.log(r_lmn) - np.log(theta)

    def s_lmn(self, l: np.ndarray, m: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Shape of the PSF 3D base on a lorentzian distribution
        Parameters :
        * `sigma` : standard deviation
        * `l,m,n` : position parameters
        """
        lorentz_hwhm = float(self._frequency_fwhm) / 2
        position_stdev = float(self._position_stdev)
        return (lorentz_hwhm**2 / (l**2 + lorentz_hwhm**2)) * np.exp(
            -(m**2 + n**2) / (2 * position_stdev**2)
        )

    def kernel(self) -> np.ndarray:
        """
        Calculation of the 3D Kernel for a gamma distribution background
        """
        L = int(self._kernel_spatial_length)
        T = int(self._kernel_frequency_length)
        m = np.arange(-L, L + 1).reshape(2 * L + 1, 1)
        n = np.arange(-L, L + 1).reshape(1, 2 * L + 1)
        M = np.repeat(m, 2 * L + 1, 1)
        N = np.repeat(n, 2 * L + 1, 0)
        O = np.arange(-T, T + 1)
        K = self._assumed_amplitude * self.s_lmn(
            O[:, np.newaxis, np.newaxis], M[np.newaxis, :], N[np.newaxis, :]
        )
        return K

    def tau(self, alpha: np.ndarray) -> np.ndarray:
        """
        Threshold calcutation function for the gamma distribution background
        Parameters:
        * `alpha`: alpha matrix
        """
        W1 = self.kernel()
        W2 = W1**2
        w1 = W1.sum()
        w2 = W2.sum()
        return np.sqrt(w2 * trigamma(alpha)) * inv_gauss(
            self._probability_false_positive
        ) + w1 * digamma(alpha)

    @property
    def settings(self) -> MoleculeDetectionSettings:
        return MoleculeDetectionSettings(
            input_file=str(self._input_file),
            quantile_gamma_fit=float(self._quantile_gamma_fit),
            quantile_max_estimate=float(self._quantile_max_estimate),
            lorentzian_safety_margin=float(self._lorentzian_safety_margin),
            probability_false_positive=float(self._probability_false_positive),
            frame_per_slice=float(self._frames_per_slice),
            frequency_fwhm=float(self._frequency_fwhm),
            position_stdev=float(self._position_stdev),
            number_of_workers=float(self._number_of_workers),
            theta=float(self._theta),
            assumed_amplitude=float(self._assumed_amplitude),
            kernel_spatial_length=float(self._kernel_spatial_length),
            kernel_frequency_length=float(self._kernel_frequency_length),
            number_of_frames_sampled_for_gamma_fitting=int(self._number_of_frames_sampled_for_gamma_fitting),
            number_of_frames_sampled_for_psf_estimation=int(self._number_of_frames_sampled_for_psf_estimation),
            border_excluded_pixels=int(self._excluded_pixels_border),
        )

    @settings.setter
    def settings(self, settings: UserActionableMoleculeDetectionSettings):
        if settings.input_file is not None:
            filepath = str(settings.input_file)
            tiff = TiffFile(filepath)
            number_of_frames = len(tiff.pages)
            self._input_file_width, self._input_file_height, = tiff.pages.first.shape
            number_of_frames_per_slice = int(self._frames_per_slice)
            number_of_slices = number_of_frames // number_of_frames_per_slice
            self._total_number_of_slices = number_of_slices
            self._number_of_frames = number_of_frames
            self.load_flattened_image(settings.input_file)
            self.load_andor_metadata(settings.input_file)
            self._input_file = settings.input_file  # ty: ignore[invalid-assignment]
        if settings.quantile_gamma_fit is not None:
            self._quantile_gamma_fit = settings.quantile_gamma_fit  # ty: ignore[invalid-assignment]
        if settings.quantile_max_estimate is not None:
            self._quantile_max_estimate = settings.quantile_max_estimate  # ty: ignore[invalid-assignment]
        if settings.lorentzian_safety_margin is not None:
            self._lorentzian_safety_margin = settings.lorentzian_safety_margin  # ty: ignore[invalid-assignment]
        if settings.probability_false_positive is not None:
            self._probability_false_positive = settings.probability_false_positive  # ty: ignore[invalid-assignment]
        if settings.frame_per_slice is not None:
            self._frames_per_slice = settings.frame_per_slice  # ty: ignore[invalid-assignment]
        if settings.number_of_frames_sampled_for_gamma_fitting is not None:
            self._number_of_frames_sampled_for_gamma_fitting = settings.number_of_frames_sampled_for_gamma_fitting  # ty: ignore[invalid-assignment]
        if settings.number_of_frames_sampled_for_psf_estimation is not None:
            self._number_of_frames_sampled_for_psf_estimation = settings.number_of_frames_sampled_for_psf_estimation  # ty: ignore[invalid-assignment]
        if settings.border_excluded_pixels is not None:
            self._excluded_pixels_border = settings.border_excluded_pixels  # ty: ignore[invalid-assignment]
        self.settings_updated.emit(self.settings)

    @Slot(str)
    def open_file(self, filepath):
        self.settings = UserActionableMoleculeDetectionSettings(input_file=filepath)

    @Slot(str)
    def open_file_and_analyze(self, filepath):
        self.settings = UserActionableMoleculeDetectionSettings(input_file=filepath)
        self.estimate_psf()
        self.estimate_from_noise()
        self.prepare_run()
        self.localize()

    @Slot()
    def analyze(self):
        if self._input_file is None:
            self.log.error("No file to analyze!")
            return
        else:
            self.open_file_and_analyze(self._input_file)

    @Slot()
    def estimate_psf(self):
        if self._input_file is None:
            self.log.error("No input file defined, could not perform noise estimation procedure.")
            return
        filepath = str(self._input_file)
        tiff = TiffFile(filepath)
        number_of_frames = len(tiff.pages)
        number_of_samples = min(self._number_of_frames_sampled_for_psf_estimation, number_of_frames)
        sampled_frames = np.random.choice(range(number_of_frames), number_of_samples)
        data = imread(
            filepath,
            key=sampled_frames,
        )
        l,m,n = np.unravel_index(np.argmax(data), data.shape)
        l = sampled_frames[l]
        extent = int(self._frequency_fwhm*self._lorentzian_safety_margin)
        sampled_frames = np.arange(max(0, l - extent), stop=min(number_of_frames, l+extent))
        extent = max(10, int(self._position_stdev*3))
        M = np.arange(max(0, m-extent), stop=min(self._input_file_width, m+extent))
        N = np.arange(max(0, n-extent), stop=min(self._input_file_height, n+extent))
        data = imread(
            filepath,
            key=sampled_frames,
        )[np.ix_(range(len(sampled_frames)), M, N)] 
        integrated_spatially = self.integrate_spatially(data)
        fitted_frequency = self.fit_frequency_dimension(integrated_spatially)
        l = int(fitted_frequency["center"])
        print(l)
        integrated_spectrally = self.integrate_spectrally(l, data)
        fitted_spatial = self.fit_spatial_dimension(integrated_spectrally)
        self._frequency_fwhm = int(np.ceil(fitted_frequency['sigma']))
        self._kernel_frequency_length = self._lorentzian_safety_margin*self._frequency_fwhm
        self._position_stdev = int(np.ceil(max(fitted_spatial['sigma_x'], fitted_spatial['sigma_x'])))
        self._kernel_spatial_length = 3*self._position_stdev

    @Slot()
    def estimate_from_noise(self):
        if self._input_file is None:
            self.log.error("No input file defined, could not perform noise estimation procedure.")
            return
        filepath = str(self._input_file)
        tiff = TiffFile(filepath)
        number_of_frames = len(tiff.pages)
        number_of_samples = min(self._number_of_frames_sampled_for_gamma_fitting, number_of_frames)
        sampled_frames = np.random.choice(range(number_of_frames), number_of_samples)
        data = np.ravel(imread(
            filepath,
            key=sampled_frames,
        ))
        self._noise_fitting_histogram = np.histogram(data, bins="auto", density=True)
        self.gamma_estimate_histogram_updated.emit(self._noise_fitting_histogram[0], self._noise_fitting_histogram[1])
        data_gamma_fit = data[ data < np.quantile(data, float(self._quantile_gamma_fit)) ]
        alpha, loc, theta = gamma.fit(data_gamma_fit, floc=0.0)
        self._theta = theta
        background = alpha * theta
        threshold_max = np.quantile(data, float(self._quantile_max_estimate))
        self._assumed_amplitude = (
            np.max(data[data < threshold_max]) - background
        )
        self.settings_updated.emit(self.settings)
        histogram_edges = self._noise_fitting_histogram[1]
        x = np.linspace(start=np.min(histogram_edges), stop=np.max(histogram_edges), num=1_000)
        y = gamma.pdf(x, alpha, loc=loc, scale=theta)
        self._fitted_gamma_law = (x, y)
        self.gamma_law_fit_updated.emit(x, y)

    def prepare_run(self):
        if self._input_file is None:
            self.log.error("Cannot prepare run, no file selected.")
            return
        self._background_kernel = self.background_kernel()
        self._workers = [
            Worker(
                initialization=self.worker_init,
                loop=self.worker, 
                finish = self.worker_finish,
                worker_id=i, thread_id=i) for i in range(int(self._number_of_workers))
        ]
        self._slicer = Worker(
                initialization=self.slicer_init,
                loop = self.slicer,
                finish = self.slicer_finish
        )
        self._builder = Worker(
                initialization=self.builder_init,
                loop=self.builder,
                finish=self.builder_finish,
        )
        for worker in self._workers:
            worker.signals.errored.connect(self.worker_thread_error, QtCore.Qt.QueuedConnection)
            worker.signals.success.connect(self.worker_thread_success, QtCore.Qt.QueuedConnection)
            worker.signals.finished.connect(self.worker_thread_finished, QtCore.Qt.QueuedConnection)
            self._trigger_interrupt.connect(worker.signals.interrupt, QtCore.Qt.QueuedConnection)
        self._slicer.signals.errored.connect(self.slicer_thread_error, QtCore.Qt.QueuedConnection)
        self._slicer.signals.success.connect(self.slicer_thread_success, QtCore.Qt.QueuedConnection)
        self._slicer.signals.finished.connect(self.slicer_thread_finished, QtCore.Qt.QueuedConnection)
        self._trigger_interrupt.connect(self._slicer.signals.interrupt)
        self._builder.signals.errored.connect(self.builder_thread_error, QtCore.Qt.QueuedConnection)
        self._builder.signals.success.connect(self.builder_thread_success, QtCore.Qt.QueuedConnection)
        self._builder.signals.finished.connect(self.builder_thread_finished, QtCore.Qt.QueuedConnection)
        self._trigger_interrupt.connect(self._builder.signals.interrupt)
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = [False for _ in self._workers]
        self._slicer_started = False
        self._builder_started = False
        self._workers_started = False
        self._total_number_of_slices = 0
        self._slicer_to_worker_queue = queue.Queue(maxsize=int(self._number_of_workers))

        filepath = str(self._input_file)
        tiff = TiffFile(filepath)
        number_of_frames = len(tiff.pages)
        number_of_frames_per_slice = int(self._frames_per_slice)
        number_of_slices = number_of_frames // number_of_frames_per_slice
        self._input_file_width, self._input_file_height, = tiff.pages.first.shape
        self._total_number_of_slices = number_of_slices
        self._number_of_frames = number_of_frames
        self._worker_to_builder_queue = queue.Queue()

    @Slot()
    def __start_watchdog_timer(self):
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.check_termination, QtCore.Qt.QueuedConnection)
        self._timer.start(1000)

    @Slot()
    def cleanup_run(self):
        self._interrupt = False
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = []
        for worker in self._workers:
            worker.signals.errored.disconnect(self.worker_thread_error)
            worker.signals.success.disconnect(self.worker_thread_success)
            worker.signals.finished.disconnect(self.worker_thread_finished)
            self._trigger_interrupt.disconnect(worker.signals.interrupt)
        self._slicer.signals.errored.disconnect(self.slicer_thread_error)
        self._slicer.signals.success.disconnect(self.slicer_thread_success)
        self._slicer.signals.finished.disconnect(self.slicer_thread_finished)
        self._trigger_interrupt.disconnect(self._slicer.signals.interrupt)
        self._builder.signals.errored.disconnect(self.builder_thread_error)
        self._builder.signals.success.disconnect(self.builder_thread_success)
        self._builder.signals.finished.disconnect(self.builder_thread_finished)
        self._trigger_interrupt.disconnect(self._builder.signals.interrupt)
        self.module_state.unlock()

    def background_kernel(self) -> np.ndarray:
        """
        Produce a "holed" kernel, i.e. a 3D matrix where the central block is zero,
        that can be used to estimate locally the background.
        """
        kf_length = int(self._kernel_frequency_length)
        ks_length = int(self._kernel_spatial_length)
        size_lateral = 2 * ks_length + 1
        size_frequency = 2 * kf_length + 1
        kernel = np.ones((size_frequency, size_lateral, size_lateral))
        kernel[np.ix_(
                    range((kf_length // 2 + 1), (kf_length + kf_length // 2 + 1)),
                    range((ks_length // 2 + 1), (ks_length + ks_length // 2 + 1)),
                    range((ks_length // 2 + 1), (ks_length + ks_length // 2 + 1)),
        )] = 0
        return kernel / np.sum(kernel)

    def adaptative_background(self, r_lmn: np.ndarray) -> np.ndarray:
        """
        Estimate the local background pixel per pixel.
        Parameters:
        * `r_lmn`: The raw data matrix.
        """
        return convolve(r_lmn, self._background_kernel, mode="valid")

    def thresholding(
        self, r_lmn: np.ndarray, background: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3D thresholding algorithm using the given `background`.
        Parameters:
        * `r_lmn` : 3D matrix given to get its thresholding 3D matrix, already filtered.
        * `background`: 3D matrix of the estimated background.

        Return a tuple of three 1D Numpy arrays containing the coordinates of
        above threshold pixels.
        """
        theta = float(self._theta)
        alpha = (
            background / theta
        )  # recover the alpha parameter of the gamma distribution of the background
        threshold = self.tau(alpha)
        thresholded = r_lmn >= threshold
        L = np.arange(thresholded.shape[0])
        M = np.arange(int(self._excluded_pixels_border), stop=thresholded.shape[1]-self._excluded_pixels_border)
        N = np.arange(int(self._excluded_pixels_border), stop=thresholded.shape[2]-self._excluded_pixels_border)
        coordinates = np.nonzero(thresholded[np.ix_(L,M,N)])
        return coordinates

    def local_maxima(
        self, r_lmn: np.ndarray, coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify the local maxima of the filtered matrix `r_lmn` given the list of
        above threshold pixels in `coordinates`.
        Parameters:
        * `r_lmn`: The filtered matrix
        * `coordinates` : coordinates of the potential maxima

        Return a 2D Numpy array in the same format as `coordinates`: three columns,
        each of which is the coordinate of a local maximum.
        """
        number_of_candidates = len(coordinates[0])
        potential_maxima_values = r_lmn[coordinates[0], coordinates[1], coordinates[2]]
        permutation = np.argsort(potential_maxima_values)[::-1]
        L = coordinates[0][permutation]
        M = coordinates[1][permutation]
        N = coordinates[2][permutation]
        is_local_maximum = np.full(number_of_candidates, LocalMaximumDecision.Undecided)
        for i in range(number_of_candidates):
            if is_local_maximum[i] != LocalMaximumDecision.Undecided:
                continue
            # Get the coordinates of the candidate being considered.
            l, m, n = L[i], M[i], N[i]
            # Compute the distance to all the other coordinates and select the
            # coordinates in the neighborhood
            spatial_distances = np.sqrt((m - M) ** 2 + (n - N) ** 2)
            roi_distances = (0 < spatial_distances) & (spatial_distances < 3 * float(self._position_stdev))
            frequency_distances = np.abs(l - L)
            roi_frequencies = (0 < frequency_distances) & (frequency_distances < 3 * float(self._frequency_fwhm))
            roi = roi_distances & roi_frequencies
            neighbors_L = L[roi]
            neighbors_M = M[roi]
            neighbors_N = N[roi]
            # Test neighborhood for maximality
            is_greater_than_neighbor = (
                r_lmn[l, m, n] > r_lmn[neighbors_L, neighbors_M, neighbors_N]
            )
            is_greater_than_neighborhood = np.all(is_greater_than_neighbor)
            # Store the result and make a decision for all neighbors (if there
            # is already a local maximum in the neighborhood, they cannot be
            # local maxima themselves!)
            if is_greater_than_neighborhood:
                is_local_maximum[roi] = LocalMaximumDecision.Discard
                is_local_maximum[i] = LocalMaximumDecision.Keep
            else:
                is_local_maximum[i] = LocalMaximumDecision.Discard
        roi_local_maxima = is_local_maximum == LocalMaximumDecision.Keep
        return (L[roi_local_maxima], M[roi_local_maxima], N[roi_local_maxima])

    def offset_result(
            self, r_lmn: np.ndarray, coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Offset the computed coordinates to account for the change in dimension due
        to the convolution steps.
        """
        data_shape = r_lmn.shape
        offset_l = (self._total_number_of_slices - data_shape[0]) // 2
        offset_m = (self._input_file_width - data_shape[1]) // 2
        offset_n = (self._input_file_height - data_shape[2]) // 2
        return (coordinates[0]+offset_l, coordinates[1]+offset_m, coordinates[2]+offset_n)

    def slice(self, slice_number: int) -> np.ndarray:
        filepath = str(self._input_file)
        number_of_frames_per_slice = int(self._frames_per_slice)
        stop_frame = min(
            self._number_of_frames, (slice_number + 1) * number_of_frames_per_slice
        )
        segment = imread(
            filepath,
            key=range(slice_number * number_of_frames_per_slice, stop_frame),
        )
        return segment
    def slicer_init(self):
        self.log.debug(f"[slicer] Starting to slice file {self._input_file}.")
        return {"slice_number": 0}

    def slicer(self, slice_number):
        """
        Slice the video and push the sliced parts in a queue to be processed by
        the workers
        """
        if self._interrupt:
            return False
        retry = True
        self.log.debug(f"[slicer] sending slice {slice_number}.")
        while retry:
            segment = self.slice(slice_number)
            try:
                self._slicer_to_worker_queue.put(
                    WorkerMessage(segment, slice_number),
                    timeout=1,
                )
            except Full:
                if not self._workers_started or all(self._worker_finished):
                    self.log.error("[slicer] All workers seem to be dead, giving up.")
                    retry = False
            else:
                retry = False
                self._start_workers.emit()
            retry = retry and not self._interrupt
        slice_number += 1
        keep_going = slice_number < self._total_number_of_slices
        return keep_going and not self._interrupt, {"slice_number":slice_number}

    def slicer_finish(self, slice_number):
        if not self._interrupt:
            self.log.debug(f"[slicer] Finished to slice file {self._input_file}.")

    def detection_procedure(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the molecule detection algorithm on the given data.
        """
        r_lmn = self.ln_transform(data)
        kernel = self.kernel()
        r_lmn = convolve(r_lmn, kernel, mode='valid')
        background = self.adaptative_background(data)
        above_threshold_coordinates = self.thresholding(r_lmn, background)
        coordinates = self.local_maxima(r_lmn, above_threshold_coordinates)
        coordinates = self.offset_result(r_lmn, coordinates)
        return coordinates

    def worker_init(self, worker_id):
        self.log.debug(f"[worker {worker_id}] Started.")
        return {"worker_id":worker_id}

    def worker(self, worker_id):
        """
        Process the slices received from the slicer and output molecule coordinates
        to a queue to be assembled back by the builder.
        """
        running = True
        try:
            message = self._slicer_to_worker_queue.get(timeout=1)
        except Empty:
            self.log.debug(f"[worker {worker_id}] Timed out!")
            running = self._slicer_started and not self._slicer_finished
        else:
            slice_number = message.slice_number
            self.log.debug(f"[worker {worker_id}] Starting to process slice {slice_number}/{self._total_number_of_slices}.")
            coordinates = self.detection_procedure(message.data)
            self._worker_to_builder_queue.put(
                BuilderMessage(coordinates, slice_number)
            )
            self._start_builder.emit()
        return running, {"worker_id": worker_id}

    def worker_finish(self, worker_id):
        self.log.debug(f"[worker {worker_id}] Finished.")

    def assemble_workers_positions(self, L, M, N):
        L = np.concat(L)
        M = np.concat(M)
        N = np.concat(N)
        coordinates = np.array([L, M, N])
        self._last_result = MoleculePositionsResult(
            file_name=str(self._input_file),
            file_metadata=self._metadata,  # ty: ignore[invalid-argument-type]
            algorithm_settings=self.settings,
            coordinates_array=coordinates,
        )
        self.new_result.emit(self._last_result)

    def builder_init(self):
        self.log.debug("[builder] Starting builder.")
        return {
                "number_of_slices_processed" : 0,
                "L" : [],
                "M" : [],
                "N" : [],
                }

    def builder(self, number_of_slices_processed, L, M, N):
        """
        Assemble the coordinates generated by the workers to produce the finale result.
        """
        running = True
        try:
            message = self._worker_to_builder_queue.get(timeout=1)
        except Empty:
            running = self._workers_started and not all(self._worker_finished)
        else:
            slice_number = message.slice_number
            slice_L, slice_M, slice_N = message.coordinates
            slice_L += slice_number * self._frames_per_slice
            L.append(slice_L)
            M.append(slice_M)
            N.append(slice_N)
            number_of_slices_processed += 1
            self.assemble_workers_positions(L, M, N)
            self.progress_update.emit(
                (number_of_slices_processed + 1) / self._total_number_of_slices
            )
        finally:
            running = running and not self._interrupt
        return running, {"number_of_slices_processed":number_of_slices_processed, "L":L, "M":M, "N":N}

    def builder_finish(self, number_of_slices_processed, L, M, N):
        self.log.debug(
            f"[builder] Processed {number_of_slices_processed}/{self._total_number_of_slices} slices."
        )
        if len(L) == 0 or len(M) == 0 or len(N) == 0:
            self.log.warning("[builder] No molecule was detected.")
            return
        self.assemble_workers_positions(L, M, N)

    @Slot()
    def worker_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"[supervisor] Worker thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def worker_thread_success(self, thread_id):
        self.log.debug(f"[supervisor] Worker {thread_id} succeeded.")

    @Slot()
    def worker_thread_finished(self, thread_id):
        self._worker_finished[thread_id] = True
        self.log.debug(f"[supervisor] Worker thread {thread_id} finished.")
        self.check_termination()

    @Slot()
    def slicer_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"[supervisor] Slicer thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def slicer_thread_success(self, _thread_id):
        self.log.debug("[supervisor] Slicer succeeded.")

    @Slot()
    def slicer_thread_finished(self, _thread_id):
        self._slicer_finished = True
        self.log.debug("[supervisor] Slicer thread finished.")
        self.check_termination()

    @Slot()
    def builder_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"[supervisor] Builder thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def builder_thread_success(self, _thread_id):
        self.log.debug("[supervisor] Builder succeeded.")

    @Slot()
    def builder_thread_finished(self, _thread_id):
        self._builder_finished = True
        self.log.debug("[supervisor] Builder thread finished.")
        self.check_termination()

    @Slot()
    def abort(self):
        self._interrupt = True
        self.failure.emit()

    @Slot()
    def check_termination(self):
        if self.module_state() != "locked":
            return
        finished = not self._builder_started or self._builder_finished
        finished = finished and (not self._slicer_started or self._slicer_finished)
        finished = finished and (not self._workers_started or all(self._worker_finished))
        if finished:
            self.cleanup_run()
            self.log.debug("[supervisor] Molecule detection procedure finished.")
            self.finished.emit()
        else:
            self._timer.start(1000)
        return finished

    @Slot()
    def __start_workers(self):
        if self._workers_started:
            return
        if len(self._workers) == 0:
            self.log.warning("No worker was initialized, could not start the processing task.")
        else:
            for worker in self._workers:
                self._threadpool.start(worker)
            self._workers_started = True

    @Slot()
    def __start_builder(self):
        if self._builder_started:
            return
        if self._builder is None:
            self.log.warning("Builder was not initialized properly and could not be started.")
        else:
            self._threadpool.start(self._builder)
            self._builder_started = True

    @Slot()
    def localize(self):
        self._localize.emit()

    @Slot()
    def __localize(self):
        if self.module_state() != 'idle':
            self.log.error(
                "Failed to acquire computing lock. Another task might be running, or the logic crashed."
            )
            return
        self.module_state.lock()
        self.prepare_run()
        self._threadpool = QThreadPool(self)
        if self._slicer is None:
            self.log.warning("Slicer was not initialized properly and could not be started.")
        else:
            self._threadpool.start(self._slicer)
            self._slicer_started = True

    @Slot()
    def interrupt(self):
        self._trigger_interrupt.emit()

    @Slot()
    def __interrupt(self):
        self._interrupt = True

    @property
    def last_result(self) -> MoleculePositionsResult | None:
        return self._last_result  # ty: ignore[invalid-return-type]

    def load_flattened_image(self, filepath):
        full_image = imread(filepath)
        self._flattened_image = np.max(full_image, axis=0)

    @property
    def flattened_image(self) -> np.ndarray:
        """Return a 2D image of the movie my taking the max over the whole time
        series of each pixel.
        """
        if self._flattened_image.size == 0 and self._input_file is not None:
            filepath = str(self._input_file)
            self.load_flattened_image(filepath)
        return self._flattened_image

    def load_andor_metadata(self, filepath: str | None):
        if filepath is not None:
            file = TiffFile(filepath)
            self._metadata = file.andor_metadata
        if self._metadata is None:
            self._metadata = {}
        for k in ANDOR_METADATA_KEYS:
            if k not in self._metadata:
                self._metadata[k] = np.nan

    @property
    def andor_metadata(self) -> dict[str, float]:
        """Return the Andor metadata from the input file."""
        if self._metadata is None:
            self.load_andor_metadata(self._input_file)
        return self._metadata

    def subset_for_pixel(self, l:int, m:int, n:int, frequency_size:int | None=None, spatial_size:int | None=None) -> tuple[np.ndarray, tuple[int, int, int]]:
        """
        Return the subset of the raw data around the localized pizel at
        `(l, m, n)`. And the offset that can be used to recover the global
        coordinates of the pixels within the subset.
        """
        if self._input_file is None:
            return np.array([])
        if frequency_size is None:
            frequency_size = int(self._kernel_frequency_length*self._lorentzian_safety_margin)
        if spatial_size is None:
            spatial_size = int(self._kernel_spatial_length)
        offset_l = max(0, l - frequency_size)
        keys = range(offset_l, min(l + frequency_size+1, self._number_of_frames))
        filepath = str(self._input_file)
        data = imread(filepath, key=keys)
        _d,w,h = data.shape
        offset_m = max(0, m - spatial_size)
        M = np.arange(offset_m, min(w, m + spatial_size + 1))
        offset_n = max(0, n - spatial_size)
        N = np.arange(offset_n, min(h, n + spatial_size + 1))
        return data[np.ix_(range(data.shape[0]), M, N)], (offset_l, offset_m, offset_n)

    def integrate_spatially(self, subset:np.ndarray) -> np.ndarray:
        """
        Return a 1D array from the 3D subset where the spatial cordinates
        have all been integrated.
        """
        return np.sum(subset, axis=(1,2)) / (subset.shape[1]*subset.shape[2])

    def integrate_spectrally(self, l:int, data:np.ndarray) -> np.ndarray:
        """Return a 2D array from the 3D dataset where the frequency axis has been integrated."""
        # return np.sum(data, axis=0) / data.shape[0]
        return data[l, :, :]

    def fit_frequency_dimension(self, integrated_spatially: np.ndarray) -> dict | None:
        model = Lorentzian()
        L = np.arange(integrated_spatially.size)
        try:
            params = model.estimate_peak(integrated_spatially, L)
            fit_result = model.fit(integrated_spatially, params, x=L)
        except:
            self.log.exception("Failed to fit on the frequency axis.")
            return
        else:
            return fit_result.params.valuesdict()

    def fit_spatial_dimension(self, integrated_spectrally: np.ndarray) -> dict | None:
        model = Gaussian2D()
        w,h = integrated_spectrally.shape
        M = np.arange(w)
        N = np.arange(h)
        MN = np.meshgrid(M, N, indexing='ij')
        raveled = np.ravel(integrated_spectrally)
        try:
            params = model.estimate_peak(raveled, MN)
            fit_result = model.fit(raveled, params, x=MN)
        except:
            self.log.exception("Failed to fit in the spatial dimension.")
            return
        else:
            return fit_result.params.valuesdict()

    @Slot()
    def superlocalize_all(self):
        number_of_pre_localized = self._last_result.coordinates_array.shape[1]
        for i in range(number_of_pre_localized):
            self.superlocalize_by_id(i)

    @Slot(int)
    def superlocalize_by_id(self, i:int) -> SingleMoleculeLocalizationDataset | None:
        """Perform a fitting procedure on the pre-localized molecule at index `i`."""
        if self._last_result is None:
            self.log.warning(f"Could not perform superlocalization on molecule {i}, no pre-localization data are available.")
            return None
        if not (0 <= i < self._last_result.coordinates_array.shape[1]):
            raise ValueError(f"There is no molecule number {i}")
        l,m,n = self._last_result.coordinates_array[:, i]
        superlocalized = self.superlocalize_by_position(l, m, n)
        if superlocalized is None:
            return None
        if superlocalized.fit_result_spectrally is None:
            l = superlocalized.position_rough_estimate_l
        else:
            l = superlocalized.fit_result_spectrally["center"]
        if superlocalized.fit_result_spatially is None:
            m = superlocalized.position_rough_estimate_m
            n = superlocalized.position_rough_estimate_n
        else:
            m = superlocalized.fit_result_spatially["center_x"]
            n = superlocalized.fit_result_spatially["center_y"]
        self._last_result.coordinates_array[:, i] = (l,m,n)
        self.position_updated.emit(i)
        return superlocalized

    @Slot(int, int)
    def superlocalize_by_position(self, l, m, n) -> SingleMoleculeLocalizationDataset | None:
        """Perform a fitting procedure assuming a molecule is close to the pixel position `(l,m,n)`."""
        self.log.debug(f"Fitting position around pixel position {(l,m,n)}.")
        data, offsets = self.subset_for_pixel(l, m, n)
        integrated_spatially = self.integrate_spatially(data)
        fit_result_spectral = self.fit_frequency_dimension(integrated_spatially)
        new_l, new_m, new_n = l,m,n
        offset_l, _offset_m, _offset_n = offsets
        if fit_result_spectral is not None:
            fit_result_spectral["center"] = fit_result_spectral["center"] + offset_l
            new_l = int(fit_result_spectral["center"])
        data, offsets = self.subset_for_pixel(new_l, new_m, new_n)
        integrated_spectrally = self.integrate_spectrally(new_l-offset_l, data)
        fit_result_spatial = self.fit_spatial_dimension(integrated_spectrally)
        _offset_l, offset_m, offset_n = offsets
        if fit_result_spatial is not None:
            fit_result_spatial["center_x"] = fit_result_spatial["center_x"] + offset_m
            fit_result_spatial["center_y"] = fit_result_spatial["center_y"] + offset_n
            new_m, new_n = int(fit_result_spatial["center_x"]), int(fit_result_spatial["center_y"])
        data, offsets = self.subset_for_pixel(new_l, new_m, new_n)
        integrated_spectrally = self.integrate_spectrally(new_l-offset_l, data)
        _offset_l, offset_m, offset_n = offsets
        superlocalized = SingleMoleculeLocalizationDataset(
                input_file=str(self._input_file),
                position_rough_estimate_l=l,
                position_rough_estimate_m=m,
                position_rough_estimate_n=n,
                flattened_image=integrated_spectrally,
                flattened_image_m=np.arange(int(offset_m), stop=int(offset_m+integrated_spectrally.shape[0])),
                flattened_image_n=np.arange(int(offset_n), stop=int(offset_n+integrated_spectrally.shape[1])),
                frequency_spectrum=np.array([np.arange(int(offset_l), stop=int(offset_l+integrated_spatially.shape[0])), integrated_spatially]),
                fit_result_spatially=fit_result_spatial,
                fit_result_spectrally=fit_result_spectral,
        )
        self._last_superlocalization = superlocalized
        self.new_superlocalization.emit(superlocalized)
        return superlocalized

    def draw_parameter_estimation_figure(self):
        histogram_bins, histogram_edges  = self._noise_fitting_histogram
        fitted_x, fitted_y = self._fitted_gamma_law 

        fig, ax = plt.subplots()
        bar = ax.bar(histogram_edges[:-1], histogram_bins, width=np.diff(histogram_edges), align="edge", label="data")
        line, = ax.plot(fitted_x, fitted_y, label=f"Fit, θ={self._theta}", color="black")
        ax.set_xlabel("Count")
        ax.set_ylabel("Density")
        ax.legend(handles=[bar, line], loc="best")
        return fig

    def draw_flattened_image_figure(self):
        data = self.flattened_image
        cbar_range = (np.nanmin(data), np.nanmax(data))
        si_prefix_cb = ScaledFloat(cbar_range[1]-cbar_range[0]).scale if cbar_range[1] != cbar_range[0] \
            else ScaledFloat(cbar_range[1])
        si_factor_cb = ScaledFloat(cbar_range[1]-cbar_range[0]).scale_val
        fig, ax = plt.subplots()
        image = ax.imshow(data.transpose()/si_factor_cb,
                            cmap='inferno',  # FIXME: reference the right place in qudi
                            origin='lower',
                            vmin=cbar_range[0]/si_factor_cb,
                            vmax=cbar_range[1]/si_factor_cb,
                            interpolation='none',
                            extent=(0, data.shape[0],
                                    0, data.shape[1])
                          )
        ax.set_aspect(1)
        ax.set_xlabel("X [Pixels]")
        ax.set_ylabel("Y [Pixels]")
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        cbar = plt.colorbar(image, shrink=0.8)  # , fraction=0.046, pad=0.08, shrink=0.75)
        cbar.set_label(f"Intensity [{si_prefix_cb}count]")
        cbar.ax.tick_params(which=u'both', length=0)
        if self.last_result is not None:
            coordinates = self.last_result.coordinates_array
            xs = coordinates[1, :]
            ys = coordinates[2, :]
            ax.plot(xs, ys, color="red", marker="o")
            labels = [str(i) for i in range(coordinates.shape[1])]
            for (x,y,label) in zip(xs,ys,labels):
                ax.annotate(label, (x,y))
        return fig

    def draw_spatial_superlocalization(self):
        if self._last_superlocalization is None:
            self.log.error("No superlocalization data to plot.")
            return
        data = self._last_superlocalization.flattened_image
        m = self._last_superlocalization.flattened_image_m
        n = self._last_superlocalization.flattened_image_n
        cbar_range = (np.nanmin(data), np.nanmax(data))
        si_prefix_cb = ScaledFloat(cbar_range[1]-cbar_range[0]).scale if cbar_range[1] != cbar_range[0] \
            else ScaledFloat(cbar_range[1])
        si_factor_cb = ScaledFloat(cbar_range[1]-cbar_range[0]).scale_val
        fig, ax = plt.subplots()
        image = ax.imshow(data.transpose()/si_factor_cb,
                            cmap='inferno',  # FIXME: reference the right place in qudi
                            origin='lower',
                            vmin=cbar_range[0]/si_factor_cb,
                            vmax=cbar_range[1]/si_factor_cb,
                            interpolation='none',
                            extent=(np.min(m), np.max(m),
                                    np.min(n), np.max(n))
                          )
        ax.set_aspect(1)
        ax.set_xlabel("X [Pixels]")
        ax.set_ylabel("Y [Pixels]")
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        cbar = plt.colorbar(image, shrink=0.8)  # , fraction=0.046, pad=0.08, shrink=0.75)
        cbar.set_label(f"Intensity [{si_prefix_cb}count]")
        cbar.ax.tick_params(which=u'both', length=0)
        if self._last_superlocalization.fit_result_spatially is not None:
            texts = []
            for (k,v) in self._last_superlocalization.fit_result_spatially.items():
                texts.append(f"{k}={v}")
            textstr = "\n".join(texts)
            props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.5}
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        return fig

    def draw_frequency_superlocalization(self):
        if self._last_superlocalization is None:
            self.log.error("No superlocalization data to plot.")
            return
        frame,count = self._last_superlocalization.frequency_spectrum
        fig, ax = plt.subplots()
        line, = ax.plot(frame, count, label="Data", color="black")
        if self._last_superlocalization.fit_result_spectrally is not None:
            offset = self._last_superlocalization.fit_result_spectrally["offset"]
            center = self._last_superlocalization.fit_result_spectrally["center"]
            sigma = self._last_superlocalization.fit_result_spectrally["sigma"]
            amplitude = self._last_superlocalization.fit_result_spectrally["amplitude"]
            x = np.linspace(start=np.min(frame), stop=np.max(frame), num=100)
            ax.plot(x, Lorentzian._model_function(x, offset, center, sigma, amplitude), label="Fit")
            textstr = "\n".join([f"offset={offset} counts",
                                 f"center={center} frames",
                                 f"HWHM={sigma} frames",
                                 f"amplitude={amplitude} counts"])
            props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.5}
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Count")
        ax.legend(handles=[line], loc="best")
        return fig

    def save_parameter_estimation(self, name_tag=""):
        timestamp = datetime.datetime.now()
        parameters = {
                "file": self._input_file,
                    "parameter estimation" : {
                    "quantile_gamma_fit": self._quantile_gamma_fit,
                    "number_of_frames_sampled": self._number_of_frames_sampled_for_gamma_fitting,
                    "quantile_max_estimate": self._quantile_max_estimate,
                },
                "estimated parameters":{
                    "amplitude": self._assumed_amplitude,
                    "theta": self._theta,
                }
                }
        x = self._noise_fitting_histogram[0]
        histo = np.concatenate((self._noise_fitting_histogram[1], [np.nan]))
        fitted = self._fitted_gamma_law[1]
        data = np.array([x,histo,fitted]).T
        header = ["Pixel value", "Data", "Fitted Gamma"]
        file_label = "fitted_background" + name_tag
        ds = TOMLHeaderDataStorage(root_dir=self.module_default_data_dir,
                             include_global_metadata=True)
        file_path, _, _ = ds.save_data(np.array(data).T,
                                       column_headers=header,
                                       metadata=parameters,
                                       nametag=file_label,
                                       timestamp=timestamp,
                                       notes="",
                                       column_dtypes=[float] * len(header))

        ds.save_thumbnail(self.draw_parameter_estimation_figure(), file_path=file_path.rsplit('.', 1)[0])

        self.log.debug(f'Background fitting saved to:{file_path}')
        return file_path
        

    def save_result(self, name_tag=""):
        if self._last_result is None:
            self.log.warning("No localization result to save!")
            return
        timestamp = datetime.datetime.now()
        parameters = {
                "file": self._last_result.file_name,
                "file metadata": self._last_result.file_metadata,
                "algorithm parameters" : asdict(self._last_result.algorithm_settings),
        }
        header = ["Frame number", "X (Pixel)", "Y (Pixel)"]
        file_label = "molecules_positions" + name_tag
        ds = TOMLHeaderDataStorage(root_dir=self.module_default_data_dir,
                             include_global_metadata=True)
        file_path, _, _ = ds.save_data(np.array(self._last_result.coordinates_array).T,
                                       column_headers=header,
                                       metadata=parameters,
                                       nametag=file_label,
                                       timestamp=timestamp,
                                       notes="",
                                       column_dtypes=[float] * len(header))

        ds.save_thumbnail(self.draw_flattened_image_figure(), file_path=file_path.rsplit('.', 1)[0])

        self.log.debug(f'Molecule coordinates saved to:{file_path}')
        return file_path

    def save_superlocalization(self, dataset:SingleMoleculeLocalizationDataset|None=None, name_tag:str=""):
        if dataset is None and self._last_superlocalization is None:
            self.log.warning("No superlocalization data to save!")
            return
        elif dataset is None:
            dataset = self._last_superlocalization
        if dataset is None:
            self.log.warning("No superlocalization data to save!")
            return
        timestamp = datetime.datetime.now()
        parameters = {
                "file": dataset.input_file,
                "Initial position estimate" : {
                    "l":dataset.position_rough_estimate_l,
                    "m":dataset.position_rough_estimate_m,
                    "n":dataset.position_rough_estimate_n,
                    },
                "Spatial Fit": dataset.fit_result_spatially,
                "Spectral Fit": dataset.fit_result_spectrally
        }
        header_spatial = ""
        header_spectral = ["Frame", "Count"]
        file_label = "molecules_superlocalization" + name_tag
        ds = TOMLHeaderDataStorage(root_dir=self.module_default_data_dir,
                             include_global_metadata=True)
        file_path_spatial, _, _ = ds.save_data(dataset.flattened_image.T,
                                       column_headers=header_spatial,
                                       metadata=parameters,
                                       nametag=file_label + "_flat_image",
                                       timestamp=timestamp,
                                       notes="",
                                       column_dtypes=[float] * dataset.flattened_image.shape[0])
        file_path_spectral, _, _ = ds.save_data(dataset.frequency_spectrum.T,
                                       column_headers=header_spectral,
                                       metadata=parameters,
                                       nametag=file_label + "_spectrum",
                                       timestamp=timestamp,
                                       notes="",
                                       column_dtypes=[float, float])

        ds.save_thumbnail(self.draw_frequency_superlocalization(), file_path=file_path_spectral.rsplit('.', 1)[0])
        ds.save_thumbnail(self.draw_spatial_superlocalization(), file_path=file_path_spatial.rsplit('.', 1)[0])

        self.log.debug(f'Molecule superlocalization coordinates saved to:{file_path_spatial,file_path_spectral}')
        return file_path_spatial, file_path_spectral

