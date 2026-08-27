import queue
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import QRunnable, QThreadPool, QTimer, Signal, Slot, QObject
from qudi.core.configoption import ConfigOption
from qudi.core.module import LogicBase
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex
from scipy.signal import convolve
from scipy.special import polygamma
from scipy.stats import gamma, norm
from tifffile import TiffFile, imread


def digamma(z):
    return polygamma(0, z)


def trigamma(z):
    return polygamma(1, z)


def inv_gauss(x):
    return norm.isf(x, loc=0, scale=1)


class LocalMaximumDecision(Enum):
    Discard = -1
    Undecided = 0
    Keep = 1


class WorkerMessageLabel(Enum):
    Data = 0
    EndOfLoop = 1


@dataclass
class WorkerMessage:
    label: WorkerMessageLabel
    data: np.ndarray | None = None
    slice_number: int = 0


@dataclass
class BuilderMessage:
    label: WorkerMessageLabel
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    slice_number: int = 0

class WorkerSignals(QObject):
    success = Signal(int)  # thread_id
    errored = Signal(tuple)
    finished = Signal(int)  # thread_id

class Worker(QRunnable):
    """
    A worker thread.
    Parameters:
    * `fn`: The function which is to be run.
    * `args`: arguments to feed the function.
    * `kwargs`: keyword arguments to feed the function.
    """

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.callback = fn
        self.args = args
        self.kwargs = kwargs
        self.thread_id = kwargs.get("thread_id", 0)
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            self.callback(*self.args, **self.kwargs)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.errored.emit((exctype, value, traceback.format_exc())) 
        else:
            self.signals.success.emit(self.thread_id)
        finally:
            self.signals.finished.emit(self.thread_id)


@dataclass
class MoleculePositionsResult:
    """
    A dataclass to hold the result of the molecule detection algorithm.
    """

    file_name: str
    quantile_gamma_fit: float
    quantile_max_estimate: float
    lorentzian_safety_margin: float
    frequency_fwhm: float
    probability_false_positive: float
    psf_extension: float
    assumed_intensity_amplitude: float
    theta: float
    kernel_spatial_length: int
    kernel_frequency_length: int
    coordinates_array: np.ndarray


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
    _quantile_max_estimate = StatusVar(name="quantile_max_estimate", default=0.9999)
    _lorentzian_safety_margin = StatusVar(name="lorentzian_safety_margin", default=3)

    _frequency_fwhm = StatusVar(name="frequency_fwhm", default=2.7)
    _position_stdev = StatusVar(name="pixel_stdev", default=(1 / 2.3548) * 2.7)
    _probability_false_positive = StatusVar(
        name="probability_false_positive", default=1e-6
    )
    _theta = StatusVar(name="theta", default=0.0)
    _assumed_amplitude = StatusVar(name="assumed_amplitude", default=1000)
    _kernel_spatial_length = StatusVar(default=5)
    _kernel_frequency_length = StatusVar(default=3)
    _input_file = StatusVar(default=None)
    _frames_per_slice = StatusVar(default=100)
    _last_result = StatusVar(default=None)

    progress_update = Signal(float)  # Localization progress between 0 and 1
    failure = Signal()
    finished = Signal()
    new_result = Signal(MoleculePositionsResult)
    settings_update = Signal(MoleculeDetectionSettings)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slicer_to_worker_queue = queue.Queue()
        self._worker_to_builder_queue = queue.Queue()
        self._background_kernel = np.array([])
        self._computing_lock = Mutex()
        self._threadpool = QThreadPool(self)
        self._workers = []
        self._slicer = Worker(self.slicer)
        self._builder = Worker(self.builder)
        self._total_number_of_slices = 0
        self._interrupt = False
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = []
        self._timer = QTimer(parent=self)

    def on_activate(self) -> None:
        pass

    def on_deactivate(self) -> None:
        # stuff
        pass

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

    def estimate_amplitude(self, mat_full_image: np.ndarray):
        """
        Estimate some parameters of the algorithm given some input data.
        """
        raveled_mat = np.ravel(mat_full_image[0])
        raveled_mat_bg = raveled_mat[
            raveled_mat < np.quantile(mat_full_image, float(self._quantile_gamma_fit))
        ]
        alpha, _loc, theta = gamma.fit(raveled_mat_bg, floc=0.0)
        background = alpha * theta
        threshold_max = np.quantile(mat_full_image, float(self._quantile_max_estimate))
        self._assumed_amplitude = (
            np.max(mat_full_image[mat_full_image < threshold_max]) - background
        )

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
        )

    @settings.setter
    def settings(self, settings: UserActionableMoleculeDetectionSettings):
        if settings.input_file is not None:
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

    def prepare_run(self):
        self._background_kernel = self.background_kernel()
        self._workers = [
            Worker(self.worker, i) for i in range(int(self._number_of_workers))
        ]
        self._slicer = Worker(self.slicer)
        self._builder = Worker(self.builder)
        for worker in self._workers:
            worker.signals.errored.connect(self.worker_thread_error)
            worker.signals.success.connect(self.worker_thread_success)
            worker.signals.finished.connect(self.worker_thread_finished)
        self._slicer.signals.errored.connect(self.slicer_thread_error)
        self._slicer.signals.success.connect(self.slicer_thread_success)
        self._slicer.signals.finished.connect(self.slicer_thread_finished)
        self._builder.signals.errored.connect(self.builder_thread_error)
        self._builder.signals.success.connect(self.builder_thread_success)
        self._builder.signals.finished.connect(self.builder_thread_finished)
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = [False for _ in self._workers]
        self._total_number_of_slices = 0
        self._slicer_to_worker_queue = queue.Queue(maxsize=int(self._number_of_workers))
        self._worker_to_builder_queue = queue.Queue()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.check_termination, QtCore.Qt.QueuedConnection)
        self._timer.start(1000)

    @Slot()
    def cleanup_run(self):
        self._interrupt = False
        self._computing_lock.release()
        self._slicer_finished = False
        self._builder_finished = False
        self._worker_finished = []
        for worker in self._workers:
            worker.signals.errored.disconnect(self.worker_thread_error)
            worker.signals.success.disconnect(self.worker_thread_success)
            worker.signals.finished.disconnect(self.worker_thread_finished)
        self._slicer.signals.errored.disconnect(self.slicer_thread_error)
        self._slicer.signals.success.disconnect(self.slicer_thread_success)
        self._slicer.signals.finished.disconnect(self.slicer_thread_finished)
        self._builder.signals.errored.disconnect(self.builder_thread_error)
        self._builder.signals.success.disconnect(self.builder_thread_success)
        self._builder.signals.finished.disconnect(self.builder_thread_finished)

    def background_kernel(self) -> np.ndarray:
        """
        Produce a "holed" kernel, i.e. a 3D matrix where the central block is zero,
        that can be used to estimate locally the background.
        """
        size_lateral = 2 * int(self._kernel_spatial_length) + 1
        size_frequency = 2 * int(self._kernel_frequency_length) + 1
        kernel = np.ones((size_frequency, size_lateral, size_lateral))
        kernel[
            (size_frequency // 2 + 1) : (size_frequency + size_frequency // 2 + 1),
            (size_lateral // 2 + 1) : (size_lateral + size_lateral // 2 + 1),
            (size_lateral // 2 + 1) : (size_lateral + size_lateral // 2 + 1),
        ] = 0
        return kernel / np.sum(kernel)

    def adaptative_background(self, r_lmn: np.ndarray) -> np.ndarray:
        """
        Estimate the local background pixel per pixel.
        Parameters:
        * `r_lmn`: The filtered matrix.
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
        coordinates = np.nonzero(thresholded)
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
            roi_distances = 0 < spatial_distances < 3 * float(self._position_stdev)
            frequency_distances = np.abs(l - L)
            roi_frequencies = 0 < frequency_distances < 3 * float(self._frequency_fwhm)
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

    def slicer(self):
        """
        Slice the video and push the sliced parts in a queue to be processed by
        the workers
        """
        filepath = str(self._input_file)
        tiff = TiffFile(filepath)
        number_of_frames = len(tiff.pages)
        self.log.debug(f"[slicer] Starting to slice file {self._input_file}.")
        number_of_frames_per_slice = int(self._frames_per_slice)
        number_of_slices = number_of_frames // number_of_frames_per_slice
        self._total_number_of_slices = number_of_slices
        for slice_number in range(number_of_slices):
            if self._interrupt:
                break
            stop_frame = min(
                number_of_frames, (slice_number + 1) * number_of_frames_per_slice
            )
            segment = imread(
                filepath,
                key=range(slice_number * number_of_frames_per_slice, stop_frame),
            )
            try:
                self._slicer_to_worker_queue.put(
                    WorkerMessage(WorkerMessageLabel.Data, segment, slice_number),
                    timeout=30,
                )
            except Full:
                if not self._interrupt:
                    self.log.error("[slicer] Timeout on pushing the slice.")
                break
        if not self._interrupt:
            # Signal end of task to the workers
            for _ in self._workers:
                self._slicer_to_worker_queue.put(
                    WorkerMessage(WorkerMessageLabel.EndOfLoop)
                )
            self.log.debug(f"[slicer] Finished to slice file {self._input_file}.")

    def worker(self, worker_id):
        """
        Process the slices received from the slicer and output molecule coordinates
        to a queue to be assembled back by the builder.
        """
        self.log.debug(f"[worker] Worker {worker_id} started.")
        running = True
        while running:
            try:
                message = self._slicer_to_worker_queue.get(timeout=5)
            except Empty:
                self.log.warning(f"[worker] Worker {worker_id} timed out!")
                last_message_label = WorkerMessageLabel.EndOfLoop
            else:
                last_message_label = message.label
                slice_number = message.slice_number
                sliced_data = message.data
                r_lmn = self.ln_transform(sliced_data)
                background = self.adaptative_background(r_lmn)
                above_threshold_coordinates = self.thresholding(r_lmn, background)
                coordinates = self.local_maxima(r_lmn, above_threshold_coordinates)
                self._worker_to_builder_queue.put(
                    BuilderMessage(WorkerMessageLabel.Data, coordinates, slice_number)
                )
            running = last_message_label != WorkerMessageLabel.EndOfLoop
            running = running and not self._interrupt
        self._worker_to_builder_queue.put(BuilderMessage(WorkerMessageLabel.EndOfLoop))
        self.log.debug(f"[worker] Worker {worker_id} finished.")

    def builder(self):
        """
        Assemble the coordinates generated by the workers to produce the finale result.
        """
        number_of_workers_finished = 0
        number_of_slices_processed = 0
        running = True
        L = []
        M = []
        N = []
        self.log.debug("[builder] Starting builder.")
        while running:
            try:
                message = self._worker_to_builder_queue.get(timeout=30)
            except Empty:
                running = False
                self.log.warning("[builder] timeout while awaiting a message")
            else:
                if message.label == WorkerMessageLabel.EndOfLoop:
                    number_of_workers_finished += 1
                    running = number_of_workers_finished < len(self._workers)
                else:
                    slice_number = message.slice_number
                    slice_L, slice_M, slice_N = message.coordinates
                    slice_L += slice_number * self._frames_per_slice
                    L.append(slice_L)
                    M.append(slice_M)
                    N.append(slice_N)
                    number_of_slices_processed += 1
                    self.progress_update.emit(
                        (number_of_slices_processed + 1) / self._total_number_of_slices
                    )
                running = running and not self._interrupt

        self.log.debug(
            f"[builder] Processed {number_of_slices_processed}/{self._total_number_of_slices} slices."
        )
        L = np.concat(L)
        M = np.concat(M)
        N = np.concat(N)
        coordinates = np.array([L, M, N])
        self._last_result = MoleculePositionsResult(
            quantile_gamma_fit=float(self._quantile_gamma_fit),
            quantile_max_estimate=float(self._quantile_max_estimate),
            lorentzian_safety_margin=float(self._lorentzian_safety_margin),
            file_name=str(self._input_file),
            frequency_fwhm=float(self._frequency_fwhm),
            probability_false_positive=float(self._probability_false_positive),
            psf_extension=float(self._position_stdev),
            assumed_intensity_amplitude=float(self._assumed_amplitude),
            theta=float(self._theta),
            kernel_spatial_length=int(self._kernel_spatial_length),
            kernel_frequency_length=int(self._kernel_frequency_length),
            coordinates_array=coordinates,
        )
        self.new_result.emit(self._last_result)

    @Slot()
    def worker_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"Worker thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def worker_thread_success(self, thread_id):
        self.log.debug(f"Worker {thread_id} succeeded.")

    @Slot()
    def worker_thread_finished(self, thread_id):
        self._worker_finished[thread_id] = True

    @Slot()
    def slicer_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"Slicer thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def slicer_thread_success(self, _thread_id):
        self.log.debug("Slicer succeeded.")

    @Slot()
    def slicer_thread_finished(self, _thread_id):
        self._slicer_finished = True

    @Slot()
    def builder_thread_error(self, error: tuple):
        exctype, value, formatted = error
        self.log.error(f"Builder thread errored {exctype}({value}): {formatted}.")
        self.abort()

    @Slot()
    def builder_thread_success(self, _thread_id):
        self.log.debug("Builder succeeded.")

    @Slot()
    def builder_thread_finished(self, _thread_id):
        self._builder_finished = True

    @Slot()
    def abort(self):
        self._interrupt = True
        self.failure.emit()

    @Slot()
    def check_termination(self):
        finished = self._builder_finished
        finished = finished and self._slicer_finished
        finished = finished and all(self._worker_finished)
        if finished:
            self.cleanup_run()
            self.finished.emit()
        else:
            self._timer.start(1000)

    @Slot()
    def localize(self):
        if not self._computing_lock.acquire(timeout=5):
            self.log.error(
                "Failed to acquire computing lock. Another task might be running, or the logic crashed."
            )
            return
        self.prepare_run()
        self._threadpool = QThreadPool(self)
        self._threadpool.start(self._slicer)
        for worker in self._workers:
            self._threadpool.start(worker)
        self._threadpool.start(self.builder)
