from PySide2 import QtCore
from PySide2 import QtWidgets

from qudi.util.widgets.scientific_spinbox import ScienDSpinBox

class CateyeScanSettingsWidget(QtWidgets.QWidget):
    """Allows choosing a configuration for the cateye laser settings."""
    sigNewScan = QtCore.Signal()
    sigDelScan = QtCore.Signal()
    sigStartScan = QtCore.Signal()
    sigSetCurrentScan = QtCore.Signal(int)
    sigSetScanName = QtCore.Signal(str)
    sigSetCalibration = QtCore.Signal(bool)
    sigNewStep = QtCore.Signal()
    sigDelStep = QtCore.Signal(int)
    sigMoveStepUp = QtCore.Signal(int)
    sigMoveStepDown = QtCore.Signal(int)
    sigSetStepGrating    = QtCore.Signal(int, int)
    sigSetStepSpan       = QtCore.Signal(int, float)
    sigSetStepOffset     = QtCore.Signal(int, float)
    sigSetStepFrequency  = QtCore.Signal(int, float)
    sigSetStepBias       = QtCore.Signal(int, float)
    sigSetStepSampleTime = QtCore.Signal(int, float)
    sigSetStepRepeat     = QtCore.Signal(int, int)
    def __init__(self, *args, **kwargs):
        super(CateyeScanSettingsWidget, self).__init__(*args, **kwargs)
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)

        scan_choice_group_box = QtWidgets.QGroupBox("Scan choice")
        main_layout.addWidget(scan_choice_group_box)
        scan_edit_group_box = QtWidgets.QGroupBox("Current scan")
        main_layout.addWidget(scan_edit_group_box)
        step_edit_group_box = QtWidgets.QGroupBox("Current step")
        main_layout.addWidget(step_edit_group_box)

        # Scan choice
        scan_choice_layout = QtWidgets.QVBoxLayout()
        scan_choice_group_box.setLayout(scan_choice_layout)
        scan_choice_controls_layout = QtWidgets.QHBoxLayout()
        self.new_scan_button = QtWidgets.QPushButton("New Scan")
        self.new_scan_button.clicked.connect(self.sigNewScan.emit)
        scan_choice_controls_layout.addWidget(self.new_scan_button)
        self.del_scan_button = QtWidgets.QPushButton("Delete current Scan")
        self.del_scan_button.clicked.connect(self.sigDelScan.emit)
        scan_choice_controls_layout.addWidget(self.del_scan_button)
        self.start_scan_button = QtWidgets.QPushButton("Start current Scan")
        self.start_scan_button.clicked.connect(self.sigStartScan.emit)
        scan_choice_controls_layout.addWidget(self.start_scan_button)
        scan_choice_layout.addLayout(scan_choice_controls_layout)
        self.all_scans_list_widget = QtWidgets.QListWidget()
        scan_choice_layout.addWidget(self.all_scans_list_widget)
        self.all_scans_list_widget.currentRowChanged.connect(self.sigSetCurrentScan.emit)
        # Scan edit
        scan_edit_layout = QtWidgets.QVBoxLayout()
        scan_edit_group_box.setLayout(scan_edit_layout)
        scan_edit_form_layout = QtWidgets.QFormLayout()
        self.scan_name_line_edit = QtWidgets.QLineEdit()
        self.scan_creation_date = QtWidgets.QLabel()
        self.calibration_check_box = QtWidgets.QCheckBox()
        self.scan_name_line_edit.editingFinished.connect(self.sigSetScanName.emit)
        self.calibration_check_box.stateChanged.connect(self.sigSetCalibration.emit)
        scan_edit_form_layout.addRow("Scan name", self.scan_name_line_edit)
        scan_edit_form_layout.addRow("Scan creation", self.scan_creation_date)
        scan_edit_form_layout.addRow("Is calibration", self.calibration_check_box)
        scan_edit_layout.addLayout(scan_edit_form_layout)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        scan_edit_layout.addWidget(line)
        step_choice_controls = QtWidgets.QHBoxLayout()
        self.new_step_button = QtWidgets.QPushButton("New step")
        self.del_step_button = QtWidgets.QPushButton("Del step")
        self.up_step_button = QtWidgets.QPushButton("Move step up")
        self.down_step_button = QtWidgets.QPushButton("Move step down")
        step_choice_controls.addWidget(self.new_step_button)
        step_choice_controls.addWidget(self.del_step_button)
        step_choice_controls.addWidget(self.up_step_button)
        step_choice_controls.addWidget(self.down_step_button)
        self.all_steps_list_widget = QtWidgets.QListWidget()
        self.new_step_button.clicked.connect(lambda : self.sigNewStep.emit)
        self.del_step_button.clicked.connect(lambda : self.sigDelStep.emit(self.all_steps_list_widget.currentRow()))
        self.up_step_button.clicked.connect(lambda : self.sigMoveStepUp.emit(self.all_steps_list_widget.currentRow()))
        self.down_step_button.clicked.connect(lambda : self.sigMoveStepDown.emit(self.all_steps_list_widget.currentRow()))
        self.all_steps_list_widget.currentRowChanged.connect(self.populate_current_step)
        # Step edit
        step_edit_layout = QtWidgets.QFormLayout()
        step_edit_group_box.setLayout(step_edit_layout)
        self.grating_spin_box = QtWidgets.QSpinBox()
        self.grating_spin_box.setMinimum(4400)
        self.grating_spin_box.setMaximum(16400)
        step_edit_layout.addRow("Grating", self.grating_spin_box)
        self.grating_spin_box.valueChanged.connect(self._emit_sig_set_step_grating)
        self.span_spin_box = QtWidgets.QDoubleSpinBox()
        self.span_spin_box.setDecimals(3)
        self.span_spin_box.setMinimum(0.0)
        self.span_spin_box.setMaximum(1.0)
        step_edit_layout.addRow("Span", self.span_spin_box)
        self.span_spin_box.valueChanged.connect(self._emit_sig_set_step_span)
        self.offset_spin_box = QtWidgets.QDoubleSpinBox()
        self.offset_spin_box.setDecimals(3)
        self.offset_spin_box.setMinimum(0.0)
        self.offset_spin_box.setMaximum(1.0)
        step_edit_layout.addRow("Offset", self.offset_spin_box)
        self.offset_spin_box.valueChanged.connect(self._emit_sig_set_step_offset)
        self.frequency_spin_box = QtWidgets.QDoubleSpinBox()
        self.frequency_spin_box.setDecimals(3)
        self.frequency_spin_box.setMinimum(0.001)
        self.frequency_spin_box.setMaximum(100.0)
        self.frequency_spin_box.setSuffix(" Hz")
        step_edit_layout.addRow("Frequency", self.frequency_spin_box)
        self.frequency_spin_box.valueChanged.connect(self._emit_sig_set_step_frequency)
        self.bias_spin_box = QtWidgets.QDoubleSpinBox()
        self.bias_spin_box.setDecimals(3)
        self.bias_spin_box.setMinimum(0.0)
        self.bias_spin_box.setMaximum(100.0)
        self.bias_spin_box.setSuffix(" mA")
        step_edit_layout.addRow("Bias", self.bias_spin_box)
        self.bias_spin_box.valueChanged.connect(self._emit_sig_set_step_bias)
        self.sample_time_spin_box = ScienDSpinBox()
        self.sample_time_spin_box.setDecimals(3)
        self.sample_time_spin_box.setMinimum(1e-4)
        self.sample_time_spin_box.setMaximum(100.0)
        self.sample_time_spin_box.setSuffix("s")
        step_edit_layout.addRow("Sample time", self.sample_time_spin_box)
        self.sample_time_spin_box.valueChanged.connect(self._emit_sig_set_step_sample_time)
        self.number_repeat_spin_box = QtWidgets.QSpinBox()
        self.number_repeat_spin_box.setMinimum(1)
        self.number_repeat_spin_box.setMaximum(100)
        step_edit_layout.addRow("Number of repeat", self.number_repeat_spin_box)
        self.number_repeat_spin_box.valueChanged.connect(self._emit_sig_set_step_repeat)

    def set_scan_list(self, scan_names):
        self.blockSignals(True)
        self.all_scans_list_widget.clear()
        self.all_scans_list_widget.addItems(scan_names)
        self.blockSignals(False)
    def set_current_scan(self, scan_index, scan):
        old_scan = self.all_scans_list_widget.currentRow()
        old_step = self.all_steps_list_widget.currentRow()
        self.blockSignals(True)
        self.all_scans_list_widget.setCurrentRow(scan_index)
        self.all_steps_list_widget.clear()
        for i in range(len(scan)):
            step = scan[i]
            item = QtWidgets.QListWidgetItem(self.generate_step_name(step))
            item.setData(QtCore.Qt.UserRole, step)
            self.all_steps_list_widget.addItem(item)
        self.scan_name_line_edit.setText(scan.name)
        self.scan_creation_date.setText(str(scan.date_created))
        self.calibration_check_box.setChecked(scan.calibration)
        self.blockSignals(False)
        if old_scan == scan_index:
            self.all_steps_list_widget.setCurrentRow(old_step)
        else:
            self.all_steps_list_widget.setCurrentRow(0)
    def generate_step_name(self, step_dict):
        return f"#{step_dict['step_no']}: grating {step_dict['grating']}, offset {step_dict['offset']}, span {step_dict['span']}, bias {step_dict['bias']}"
    def set_current_step(self, step_index):
        self.all_steps_list_widget.setCurrentRow(step_index)
    def populate_current_step(self, i):
        step = self.all_steps_list_widget.currentItem().data(QtCore.Qt.UserRole)
        self.blockSignals(True)
        self.grating_spin_box.setValue(step["grating"])
        self.span_spin_box.setValue(step["span"])
        self.offset_spin_box.setValue(step["offset"])
        self.frequency_spin_box.setValue(step["frequency"])
        self.bias_spin_box.setValue(step["bias"])
        self.sample_time_spin_box.setValue(step["sample_time"])
        self.number_repeat_spin_box.setValue(step["repeat"])
        self.blockSignals(False)
    def _emit_sig_set_step(self, sig, value):
        step = self.all_steps_list_widget.currentItem().data(QtCore.Qt.UserRole)
        sig.emit(step["step_no"], value)
    def _emit_sig_set_step_grating(self, value):
        self._emit_sig_set_step(self.sigSetStepGrating, value)
    def _emit_sig_set_step_span(self, value):
        self._emit_sig_set_step(self.sigSetStepSpan, value)
    def _emit_sig_set_step_offset(self, value):
        self._emit_sig_set_step(self.sigSetStepOffset, value)
    def _emit_sig_set_step_frequency(self, value):
        self._emit_sig_set_step(self.sigSetStepFrequency, value)
    def _emit_sig_set_step_bias(self, value):
        self._emit_sig_set_step(self.sigSetStepBias, value)
    def _emit_sig_set_step_sample_time(self, value):
        self._emit_sig_set_step(self.sigSetStepSampleTime, value)
    def _emit_sig_set_step_repeat(self, value):
        self._emit_sig_set_step(self.sigSetStepRepeat, value)
