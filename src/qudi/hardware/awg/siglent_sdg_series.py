import os
try:
    import pyvisa as visa
except ImportError:
    import visa
import numpy as np
from qudi.util.paths import get_appdata_dir
from qudi.util.helpers import natural_sort
from qudi.core.configoption import ConfigOption
from qudi.core.statusvariable import StatusVar
from qudi.interface.pulser_interface import PulserInterface, PulserConstraints, SequenceOption


class SDG6022X(PulserInterface):
    visa_address = ConfigOption(name="visa_address", missing="error")
    output1load = ConfigOption(name="output1load", default=None)
    output2load = ConfigOption(name="output2load", default=None)
    _tmp_work_dir = ConfigOption(name='tmp_work_dir',
                                 default=os.path.join(get_appdata_dir(True), 'pulsed_files'),
                                 missing='warn')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rm = visa.ResourceManager()
        self._inst = None

    def on_activate(self):
        # Create work directory if necessary
        if not os.path.exists(self._tmp_work_dir):
            os.makedirs(os.path.abspath(self._tmp_work_dir))
        try:
            self._inst = self._rm.open_resource(self.visa_address)
            self._inst.timeout = 30000
        except:
            self._inst = None
            self.log.error(f"Unable to connect to the AWG at VISA address {self.visa_address}.")
        idn = self.query("*IDN?").split(',')
        self.log.debug(f'Received response to IDN {idn}')
        if len(idn) > 4:
            self.manufacturer = "Siglent Technologies"
            self.device_id = idn[0]
            self.hardware_version = idn[4]
        else:
            self.manufacturer = idn[0]
            self.device_id = "SDG"
            self.hardware_version = None
        self.model = idn[1]
        self.serial_number = idn[2]
        self.firmware_version = idn[3]
        self.log.info(f"Connected to {self.manufacturer} {self.model} {self.device_id}\nSerial: {self.serial_number}\nFirmware: {self.firmware_version}\nHardware informations: {self.hardware_version}")
        # Constraints
        self._constraints = PulserConstraints()
        # We do not support sequences in this hardware
        self._constraints.sequence_option = SequenceOption.NON
        # Sampling rate (page 347 of the manual)
        self._constraints.sample_rate.min = 1e-6
        self._constraints.sample_rate.max = 300e6
        self._constraints.sample_rate.step = 1
        self._constraints.sample_rate.default = 300e6
        # analog signals
        self._constraints.a_ch_amplitude.min = 0.001
        self._constraints.a_ch_amplitude.max = 5.0
        self._constraints.a_ch_amplitude.step = 0.001
        self._constraints.a_ch_amplitude.default = 0.100
        self._constraints.a_ch_offset.min = -5.0
        self._constraints.a_ch_offset.max = 5.0
        self._constraints.a_ch_offset.step = 0.001
        self._constraints.a_ch_offset.default = 0.0
        # waveforms
        self._constraints.waveform_length.min = 2
        self._constraints.waveform_length.max = 2**24
        self._constraints.waveform_length.step = 1
        self._constraints.waveform_length.default = 80
        # Some configurations
        self._constraints.activation_config = {
            'all_active' : frozenset({'a_ch1', 'a_ch2'}),
            'only_chnl1' : frozenset({'a_ch1'}),
            'only_chnl2' : frozenset({'a_ch2'})
        }
        self.reset()
        self.__ch1_active = True
        self.__ch2_active = True

    def write(self, command):
        if self._inst is None:
            pass 
        self._inst.write(command)
        return 0

    def query(self, command):
        if self._inst is None:
            pass 
        return self._inst.query(command).rstrip('\n')

    def read(self):
        if self._inst is None:
            pass 
        return self._inst.read()
        
    def on_deactivate(self):
        self.pulser_off()
        self._inst.close()

    def get_constraints(self):
        """
        Retrieve the hardware constraints from the Pulsing device.

        @return constraints object: object with pulser constraints as attributes.
        """
        return self._constraints

    def pulser_on(self):
        """ Switches the pulsing device on.

        @return int: error code (0:OK, -1:error)
        """
        if self.__ch1_active:
            args = ["LOAD", str(self.output1load)] if self.output1load is not None else []
            self.write("C1:OUTPUT " + ",".join(["ON"] + args))
        if self.__ch2_active:
            args = ["LOAD", str(self.output2load)] if self.output2load is not None else []
            self.write("C2:OUTPUT " + ",".join(["ON"] + args))
        return 0
        
    def pulser_off(self):
        """ Switches the pulsing device on.

        @return int: error code (0:OK, -1:error)
        """
        self.write("C1:OUTPUT OFF")
        self.write("C2:OUTPUT OFF")
        return 0

    def load_waveform(self, load_dict):
        """ Loads a waveform to the specified channel of the pulsing device.

        @param dict|list load_dict: a dictionary with keys being one of the available channel
                                    index and values being the name of the already written
                                    waveform to load into the channel.
                                    Examples:   {1: rabi_ch1, 2: rabi_ch2} or
                                                {1: rabi_ch2, 2: rabi_ch1}
                                    If just a list of waveform names if given, the channel
                                    association will be invoked from the channel
                                    suffix '_ch1', '_ch2' etc.

                                        {1: rabi_ch1, 2: rabi_ch2}
                                    or
                                        {1: rabi_ch2, 2: rabi_ch1}

                                    If just a list of waveform names if given,
                                    the channel association will be invoked from
                                    the channel suffix '_ch1', '_ch2' etc. A
                                    possible configuration can be e.g.

                                        ['rabi_ch1', 'rabi_ch2', 'rabi_ch3']

        @return dict: Dictionary containing the actually loaded waveforms per
                      channel.
        """
        if isinstance(load_dict, list):
            new_dict = dict()
            for waveform in load_dict:
                channel = int(waveform.rsplit('_ch', 1)[1])
                new_dict[channel] = waveform
            load_dict = new_dict
        # Get all active channels
        chnl_activation = self.get_active_channels()
        analog_channels = natural_sort(
            chnl for chnl in chnl_activation if chnl.startswith('a') and chnl_activation[chnl])
        # Check if all channels to load to are active
        channels_to_set = {'a_ch{0:d}'.format(chnl_num) for chnl_num in load_dict}
        if not channels_to_set.issubset(analog_channels):
            self.log.error('Unable to load all waveforms into channels.\n'
                           'One or more channels to set are not active.')
            return self.get_loaded_assets()[0]
        # Check if all waveforms to load are present on device memory
        if not set(load_dict.values()).issubset(self.get_waveform_names()):
            self.log.error('Unable to load waveforms into channels.\n'
                           'One or more waveforms to load are missing on device memory.')
            self.log.debug(f'Load dict is {load_dict.values()}, waveform names is {self.get_waveform_names()}')
            return self.get_loaded_assets()[0]
        # Load waveforms into channels
        for chnl_num, waveform in load_dict.items():
            param = "C1" if chnl_num == 1 else "C2"
            self.write(f"{param}:OUTPUT NAME,{waveform}")
        return self.get_loaded_assets()[0]

    def load_sequence(self, sequence_name):
        return self.get_loaded_assets()[0]

    def get_loaded_assets(self):
        """
        Retrieve the currently loaded asset names for each active channel of the device.
        The returned dictionary will have the channel numbers as keys.
        In case of loaded waveforms the dictionary values will be the waveform names.
        In case of a loaded sequence the values will be the sequence name appended by a suffix
        representing the track loaded to the respective channel (i.e. '<sequence_name>_1').

        @return (dict, str): Dictionary with keys being the channel number and values being the
                             respective asset loaded into the channel,
                             string describing the asset type ('waveform' or 'sequence')
        """
        # Get all active channels
        chnl_activation = self.get_active_channels()

        channel_numbers = sorted(int(chnl.split('_ch')[1]) for chnl in chnl_activation if
                                 chnl.startswith('a') and chnl_activation[chnl])
        # Get assets per channel
        loaded_assets = dict()
        if 1 in channel_numbers:
            response = self.query("C1:ARWV?").split(",")[-1].rstrip('.bin')
            loaded_assets[1] = response
        if 2 in channel_numbers:
            response = self.query("C2:ARWV?").split(",")[-1].rstrip('.bin')
            loaded_assets[2] = response
        return loaded_assets, "waveform"

    def clear_all(self):
        """ Clears all loaded waveforms from the pulse generators RAM/workspace.
        (not implemented)

        @return int: error code (0:OK, -1:error)
        """
        return 0

    def get_status(self):
        """ Retrieves the status of the pulsing hardware

        @return (int, dict): tuple with an integer value of the current status and a corresponding
                             dictionary containing status description for all the possible status
                             variables of the pulse generator hardware.
        """
        status_dict = {
            0: "All output off",
            1: "Channel 1 on, Channel 2 off",
            2: "Channel 1 off, Channel 2 on",
            3: "Channel 1 on, Channel 2 on",
        }
        status1 = "OUTP ON" in self.query("C1:OUTPUT?")
        status2 = "OUTP ON" in self.query("C2:OUTPUT?")
        return status1 | status2 << 1, status_dict

    def get_sample_rate(self):
        """ Get the sample rate of the pulse generator hardware

        @return float: The current sample rate of the device (in Hz)
        """
        res = self.query("C1:SAMPLERATE?").split(",")
        i = res.index("VALUE")
        return float(res[i+1][:-4])

    def set_sample_rate(self, sample_rate):
        """ Set the sample rate of the pulse generator hardware.

        We force the two channels to have the same sampling rate.

        @param float sample_rate: The sampling rate to be set (in Hz)

        @return float: the sample rate returned from the device (in Hz).
        """
        self.write(f"C1:samplerate value,{sample_rate}")
        self.write(f"C2:samplerate value,{sample_rate}")
        return self.get_sample_rate()

    def get_analog_level(self, amplitude=None, offset=None):
        """ Retrieve the analog amplitude and offset of the provided channels.

        @param list amplitude: optional, if the amplitude value (in Volt peak to peak, i.e. the
                               full amplitude) of a specific channel is desired.
        @param list offset: optional, if the offset value (in Volt) of a specific channel is
                            desired.

        @return: (dict, dict): tuple of two dicts, with keys being the channel descriptor string
                               (i.e. 'a_ch1') and items being the values for those channels.
                               Amplitude is always denoted in Volt-peak-to-peak and Offset in volts.

        Note: Do not return a saved amplitude and/or offset value but instead retrieve the current
              amplitude and/or offset directly from the device.

        If nothing (or None) is passed then the levels of all channels will be returned. If no
        analog channels are present in the device, return just empty dicts.

        Example of a possible input:
            amplitude = ['a_ch1', 'a_ch4'], offset = None
        to obtain the amplitude of channel 1 and 4 and the offset of all channels
            {'a_ch1': -0.5, 'a_ch4': 2.0} {'a_ch1': 0.0, 'a_ch2': 0.0, 'a_ch3': 1.0, 'a_ch4': 0.0}
        """
        amp = dict()
        off = dict()
        res = self.query("C1:basic_wave?").split(",")
        ampindex = res.index("AMP")+1
        amplitude1 = float(res[ampindex][:-1])
        ofstindex = res.index("OFST")+1
        offset1 = float(res[ofstindex][:-1])
        res = self.query("C2:basic_wave?").split(",")
        ampindex = res.index("AMP")+1
        amplitude2 = float(res[ampindex][:-1])
        ofstindex = res.index("OFST")+1
        offset2 = float(res[ofstindex][:-1])
        # get pp amplitudes
        if amplitude is None:
            amp["a_ch1"] = amplitude1
            amp["a_ch2"] = amplitude2
        else:
            for chnl in amplitude:
                if chnl == "a_ch1":
                    amp["a_ch1"] = amplitude1
                elif chnl == "a_ch2":
                    amp["a_ch2"] = amplitude2
                else:
                    self.log.warning('Get analog amplitude channel "{0}" failed. '
                                     'Channel non-existent.'.format(chnl))

        # get voltage offsets
        if offset is None:
            off["a_ch1"] = offset1
            off["a_ch2"] = offset2
        else:
            for chnl in offset:
                if chnl == "a_ch1":
                    off["a_ch1"] = offset1
                elif chnl == "a_ch2":
                    off["a_ch2"] = offset2
                else:
                    self.log.warning('Get analog offset channel "{0}" failed. '
                                     'Channel non-existent.'.format(chnl))
        return amp, off

    def set_analog_level(self, amplitude=None, offset=None):
        """ Set amplitude and/or offset value of the provided analog channel(s).

        @param dict amplitude: dictionary, with key being the channel descriptor string
                               (i.e. 'a_ch1', 'a_ch2') and items being the amplitude values
                               (in Volt peak to peak, i.e. the full amplitude) for the desired
                               channel.
        @param dict offset: dictionary, with key being the channel descriptor string
                            (i.e. 'a_ch1', 'a_ch2') and items being the offset values
                            (in absolute volt) for the desired channel.

        @return (dict, dict): tuple of two dicts with the actual set values for amplitude and
                              offset for ALL channels.

        If nothing is passed then the command will return the current amplitudes/offsets.

        Note: After setting the amplitude and/or offset values of the device, use the actual set
              return values for further processing.
        """
        constraints = self.get_constraints()
        # amplitude sanity check
        if amplitude is not None:
            for chnl in amplitude:
                if chnl not in ("a_ch1", "a_ch2"):
                    self.log.warning('Channel to set ({0}) not available in AWG.\nSetting '
                                     'analogue voltage for this channel ignored.'.format(chnl))
                    del amplitude[chnl]
                if amplitude[chnl] < constraints.a_ch_amplitude.min:
                    self.log.warning('Minimum Vpp for channel "{0}" is {1}. Requested Vpp of {2}V '
                                     'was ignored and instead set to min value.'
                                     ''.format(chnl, constraints.a_ch_amplitude.min,
                                               amplitude[chnl]))
                    amplitude[chnl] = constraints.a_ch_amplitude.min
                elif amplitude[chnl] > constraints.a_ch_amplitude.max:
                    self.log.warning('Maximum Vpp for channel "{0}" is {1}. Requested Vpp of {2}V '
                                     'was ignored and instead set to max value.'
                                     ''.format(chnl, constraints.a_ch_amplitude.max,
                                               amplitude[chnl]))
                    amplitude[chnl] = constraints.a_ch_amplitude.max
        # offset sanity check
        if offset is not None:
            for chnl in offset:
                if chnl not in ("a_ch1", "a_ch2"):
                    self.log.warning('Channel to set (a_ch{0}) not available in AWG.\nSetting '
                                     'offset voltage for this channel ignored.'.format(chnl))
                    del offset[chnl]
                if offset[chnl] < constraints.a_ch_offset.min:
                    self.log.warning('Minimum offset for channel "{0}" is {1}. Requested offset of '
                                     '{2}V was ignored and instead set to min value.'
                                     ''.format(chnl, constraints.a_ch_offset.min, offset[chnl]))
                    offset[chnl] = constraints.a_ch_offset.min
                elif offset[chnl] > constraints.a_ch_offset.max:
                    self.log.warning('Maximum offset for channel "{0}" is {1}. Requested offset of '
                                     '{2}V was ignored and instead set to max value.'
                                     ''.format(chnl, constraints.a_ch_offset.max,
                                               offset[chnl]))
                    offset[chnl] = constraints.a_ch_offset.max
        if amplitude is not None:
            for chnl, amp in amplitude.items():
                param = "C1" if chnl == "a_ch1" else "C2"
                self.write(f"{param}:basic_wave AMP,{amp:.4f}")
        if offset is not None:
            for chnl, off in offset.items():
                param = "C1" if chnl == "a_ch1" else "C2"
                self.write(f"{param}:basic_wave OFST,{off:.4f}")
        return self.get_analog_level()

    def get_digital_level(self, low=None, high=None):
        """ Retrieve the digital low and high level of the provided/all channels.

        @param list low: optional, if the low value (in Volt) of a specific channel is desired.
        @param list high: optional, if the high value (in Volt) of a specific channel is desired.

        @return: (dict, dict): tuple of two dicts, with keys being the channel descriptor strings
                               (i.e. 'd_ch1', 'd_ch2') and items being the values for those
                               channels. Both low and high value of a channel is denoted in volts.
        """
        return {}, {}

    def set_digital_level(self, low=None, high=None):
        """ Set low and/or high value of the provided digital channel.

        @param dict low: dictionary, with key being the channel descriptor string
                         (i.e. 'd_ch1', 'd_ch2') and items being the low values (in volt) for the
                         desired channel.
        @param dict high: dictionary, with key being the channel descriptor string
                          (i.e. 'd_ch1', 'd_ch2') and items being the high values (in volt) for the
                          desired channel.

        @return (dict, dict): tuple of two dicts where first dict denotes the current low value and
                              the second dict the high value for ALL digital channels.
                              Keys are the channel descriptor strings (i.e. 'd_ch1', 'd_ch2')
        """
        return {}, {}

    def get_active_channels(self, ch=None):
        """ Get the active channels of the pulse generator hardware.

        @param list ch: optional, if specific analog or digital channels are needed to be asked
                        without obtaining all the channels.

        @return dict:  where keys denoting the channel string and items boolean expressions whether
                       channel are active or not.

        Example for an possible input (order is not important):
            ch = ['a_ch2', 'd_ch2', 'a_ch1', 'd_ch5', 'd_ch1']
        then the output might look like
            {'a_ch2': True, 'd_ch2': False, 'a_ch1': False, 'd_ch5': True, 'd_ch1': False}

        If no parameter (or None) is passed to this method all channel states will be returned.
        """
        active_ch = dict()
        output1_asked = ch is None or "a_ch1" in ch
        output2_asked = ch is None or "a_ch2" in ch
        if output1_asked:
            active_ch["a_ch1"] = self.__ch1_active
        if output2_asked:
            active_ch["a_ch2"] = self.__ch2_active
        return active_ch

    def set_active_channels(self, ch=None):
        """
        Set the active/inactive channels for the pulse generator hardware.
        The state of ALL available analog and digital channels will be returned
        (True: active, False: inactive).
        The actually set and returned channel activation must be part of the available
        activation_configs in the constraints.
        You can also activate/deactivate subsets of available channels but the resulting
        activation_config must still be valid according to the constraints.
        If the resulting set of active channels can not be found in the available
        activation_configs, the channel states must remain unchanged.

        @param dict ch: dictionary with keys being the analog or digital string generic names for
                        the channels (i.e. 'd_ch1', 'a_ch2') with items being a boolean value.
                        True: Activate channel, False: Deactivate channel

        @return dict: with the actual set values for ALL active analog and digital channels

        If nothing is passed then the command will simply return the unchanged current state.

        Note: After setting the active channels of the device, use the returned dict for further
              processing.

        Example for possible input:
            ch={'a_ch2': True, 'd_ch1': False, 'd_ch3': True, 'd_ch4': True}
        to activate analog channel 2 digital channel 3 and 4 and to deactivate
        digital channel 1. All other available channels will remain unchanged.
        """
        currently_active_channels = self.get_active_channels()
        if ch is None:
            return currently_active_channels
        if not set(currently_active_channels).issuperset(ch):
            self.log.error("Trying to (de)activate channels that are not present in the AWG.\n"
                           "Operation aborted.")
            return currently_active_channels
        if "a_ch1" in ch:
            self.__ch1_active = ch["a_ch1"]
        if "a_ch2" in ch:
            self.__ch2_active = ch["a_ch2"]
        return self.get_active_channels()

    def write_waveform(self, name, analog_samples, digital_samples, is_first_chunk, is_last_chunk,
                       total_number_of_samples):
        """
        Write a new waveform or append samples to an already existing waveform on the device memory.
        The flags is_first_chunk and is_last_chunk can be used as indicator if a new waveform should
        be created or if the write process to a waveform should be terminated.

        NOTE: All sample arrays in analog_samples and digital_samples must be of equal length!

        @param str name: the name of the waveform to be created/append to
        @param dict analog_samples: keys are the generic analog channel names (i.e. 'a_ch1') and
                                    values are 1D numpy arrays of type float32 containing the
                                    voltage samples normalized to half Vpp (between -1 and 1).
        @param dict digital_samples: keys are the generic digital channel names (i.e. 'd_ch1') and
                                     values are 1D numpy arrays of type bool containing the marker
                                     states.
        @param bool is_first_chunk: Flag indicating if it is the first chunk to write.
                                    If True this method will create a new empty wavveform.
                                    If False the samples are appended to the existing waveform.
        @param bool is_last_chunk:  Flag indicating if it is the last chunk to write.
                                    Some devices may need to know when to close the appending wfm.
        @param int total_number_of_samples: The number of sample points for the entire waveform
                                            (not only the currently written chunk)

        @return (int, list): Number of samples written (-1 indicates failed process) and list of
                             created waveform names
        """
        waveforms = []
        if len(analog_samples) == 0:
            self.log.error("No analog samples passed to write_waveform method.")
            return -1, waveforms
        if total_number_of_samples > self._constraints.waveform_length.max:
            self.log.error(f"Unable to write {total_number_of_samples} samples. The maximum allowed number of samples is {self._constraints.waveform_length.max}.")
            return -1, waveforms
        if total_number_of_samples < self._constraints.waveform_length.min:
            self.log.error(f"Unable to write {total_number_of_samples} samples. The minimum allowed number of samples is {self._constraints.waveform_length.min}.")
            return -1, waveformsz
        if len(digital_samples) > 0:
            self.log.error("Digital samples are not allowed in this AWG.")
            return -1, waveforms
        chunk_length = None
        for a_ch in analog_samples.keys():
            this_length = len(analog_samples[a_ch])
            if this_length > total_number_of_samples:
                self.log.error(f"The analog samples in channel {a_ch} are longer than the total length ({this_length} instead of {total_number_of_samples})")
                return (-1, waveforms)
            elif chunk_length is None:
                chunk_length = len(analog_samples[a_ch])
            elif this_length != chunk_length:
                self.log.error(f"The analog samples for channel {a_ch} are different from the chunk length ({this_length} instead of {chunk_length}).")
        for a_ch in analog_samples.keys():
            if a_ch == "a_ch1":
                chosen_channel = 1 
            elif a_ch == "a_ch2":
                chosen_channel = 2 
            else:
                self.log.error(f"Unknown channel \"{a_ch}\", skipping.")
                continue
            s = analog_samples[a_ch]
            mini = np.min(s)
            maxi = np.max(s)
            if mini < -1 or maxi > 1:
                s = np.clip(s, -1, 1)
                self.log.warning(f"Got waveform out of the range [-1,1], actual range is [{mini},{maxi}]. Output got clipped.")
            mini = np.iinfo(np.int16).min
            maxi = np.iinfo(np.int16).max
            s = ((s/2 + 0.5) * (maxi - mini) + mini).astype(np.int16)
            filename = f'{name}_ch{chosen_channel}'
            filepath = os.path.join(self._tmp_work_dir, filename + '.bin')
            openoptions = 'wb' if is_first_chunk else 'ab'
            with open(filepath, openoptions) as binfile:
                s.tofile(binfile)
            if is_last_chunk and self._inst is not None:
                s = np.fromfile(filepath, dtype=np.int16)
                self.log.debug(f"Loaded from {filepath}")
                self.log.debug(f"About to write a {len(s.byteswap().tobytes())} array")
                self.log.debug(f'First data is {s[0]}')
                # self._inst.write_binary_values(f"C{chosen_channel}:WVDT FREQ,1.0,AMPL,1.0,OFST,0.0,PHASE,0.0,WVNM,\"{filename}\",WAVEDATA,", s.byteswap().tobytes(), datatype='s')
                self._inst.write_binary_values(f"C{chosen_channel}:WVDT WVNM,{filename},LENGTH,{len(s)},WAVEDATA, ", s.byteswap().tobytes(), datatype='s', header_fmt='empty')
                self._inst.write(f"C{chosen_channel}:ARWV NAME,{filename}")
            waveforms.append(filename)
        return total_number_of_samples, waveforms

    def write_sequence(self, name, sequence_parameters):
        """
        Write a new sequence on the device memory.

        @param str name: the name of the waveform to be created/append to
        @param list sequence_parameters: List containing tuples of length 2. Each tuple represents
                                         a sequence step. The first entry of the tuple is a list of
                                         waveform names (str); one for each channel. The second
                                         tuple element is a SequenceStep instance containing the
                                         sequencing parameters for this step.

        @return: int, number of sequence steps written (-1 indicates failed process)
        """
        self.log.error("Sequences are note supported by this AWG.")
        return -1

    def get_waveform_names(self):
        """ Retrieve the names of all uploaded waveforms on the device.

        @return list: List of all uploaded waveform name strings in the device workspace.
        """
        return self.query("STL? USER").split(",")[1:]

    def get_sequence_names(self):
        """ Retrieve the names of all uploaded sequence on the device.

        @return list: List of all uploaded sequence name strings in the device workspace.
        """
        return []

    def delete_waveform(self, waveform_name):
        """ Delete the waveform with name "waveform_name" from the device memory.

        @param str waveform_name: The name of the waveform to be deleted
                                  Optionally a list of waveform names can be passed.

        @return list: a list of deleted waveform names.
        """
        return [waveform_name]

    def delete_sequence(self, sequence_name):
        """ Delete the sequence with name "sequence_name" from the device memory.

        @param str sequence_name: The name of the sequence to be deleted
                                  Optionally a list of sequence names can be passed.

        @return list: a list of deleted sequence names.
        """
        return []

    def get_interleave(self):
        """ Check whether Interleave is ON or OFF in AWG.

        @return bool: True: ON, False: OFF

        Will always return False for pulse generator hardware without interleave.
        """
        return False

    def set_interleave(self, state=False):
        """ Turns the interleave of an AWG on or off.

        @param bool state: The state the interleave should be set to
                           (True: ON, False: OFF)

        @return bool: actual interleave status (True: ON, False: OFF)

        Note: After setting the interleave of the device, retrieve the
              interleave again and use that information for further processing.

        Unused for pulse generator hardware other than an AWG.
        """
        return False

    def reset(self):
        """ Reset the device.

        @return int: error code (0:OK, -1:error)
        """
        self.write("*RST")
        self.write("C1:basic_wave wvtp,ARB")
        self.write("C2:basic_wave wvtp,ARB")
        self.write("C1:samplerate mode,TARB")
        self.write("C2:samplerate mode,TARB")
        self.write("eqphase on")
        args = ["LOAD", str(self.output1load)] if self.output1load is not None else []
        self.write("C1:OUTPUT " + ",".join(["OFF"] + args))
        args = ["LOAD", str(self.output2load)] if self.output2load is not None else []
        self.write("C2:OUTPUT " + ",".join(["OFF"] + args))
        return 0
