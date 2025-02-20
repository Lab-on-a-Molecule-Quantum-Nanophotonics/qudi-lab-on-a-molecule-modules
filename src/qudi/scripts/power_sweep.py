from qudi.core.scripting.moduletask import ModuleTask
import time
from datetime import datetime
import numpy as np

class PowerSweepMeasurement(ModuleTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._powers = None
        self._powers_times = None
        self._values = None
        self._times = None
        self._interpolated_powers = None
        self._power_index = 0
        self._values_index = 0
        self.channel = 'pfi1'
        self.duration = 120
        self.plot_index = 0
        self.timestamp = datetime.now()
        self.timestart = time.perf_counter()
        self.buffer_size = 2**16
        self.channel_count = 0

    def update_data(self, data, data_time):
        self._times[self._values_index:self._values_index+data_time.size] = data_time
        self._values[(self._values_index*self.channel_count):((self._values_index+data_time.size)*self.channel_count)] = data
        self._values_index += data_time.size

    def update_power(self, power, power_time):
        self._powers.append(power)
        self._powers_times.append(power_time)

    def interpolate(self):
        self._interpolated_powers = np.interp(self._times[:self._values_index], self._powers_times[:self._power_index], self._powers[:self._power_index])
        qdplot_logic.set_data(
            plot_index=self.plot_index, 
            data={
                k: (self._interpolated_powers, self.values[i::self.channel_count])
                for (i,k) in enumerate(time_series_reader_logic.active_channel_names)
            },
            name=f"Power sweep {self.timestamp}",
            clear_old=True,
        )
        
    def _setup(self):
        self.log.info("Starting power sweep.")
        qdplot_logic.add_plot()
        self.plot_index = qdplot_logic.plot_count - 1
        self.timestamp = datetime.now()
        self.timestart= time.perf_counter()
        self.channel_count = len(time_series_reader_logic.active_channel_names)
        self._powers = np.zeros(self.buffer_size)
        self._powers_times = np.zeros(self.buffer_size)
        self._values = np.zeros(self.buffer_size*self.channel_count)
        self._times = np.zeros(self.buffer_size)
        powermeter_time_series_logic.sigNewRawData.connect(self.update_power)
        time_series_reader_logic.sigNewRawData.connect(self.update_data)
        if powermeter_time_series_logic.module_state() == 'locked': 
            powermeter_time_series_logic.stop_reading()
        powermeter_time_series_logic.start_recording()
        if time_series_reader_logic.module_state() == 'locked': 
            time_series_reader_logic.stop_reading()
        time_series_reader_logic.start_recording()

    def _run(self):
        while time.perf_counter() - self.timestart < self.duration:
            time.sleep(1)
            self.interpolate()

    def _cleanup(self):
        powermeter_time_series_logic.sigNewRawData.disconnect(self.update_power)
        time_series_reader_logic.sigNewRawData.disconnect(self.update_data)
        time_series_reader_logic.stop_recording()
        powermeter_time_series_logic.stop_recording()
        self.interpolate()
        path = qdplot_logic.save_data(self.plot_index, "power_sweep")
        self.log.info(f"Power sweep saved to {path}")
