# %%
import numpy as np
import mne
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum
import joblib

matplotlib.use('TkAgg')
# %%
# Find the streams by its name
eeg_inlet_name = 'EEG'
eeg_streams_info = resolve_byprop('name', eeg_inlet_name)

markers_inlet_name = 'speller_matrix_markers_online'
marker_streams_info = resolve_byprop('name', markers_inlet_name)

# Set inlets
#If sample at 500Hz then in 1 min I will have 30000 samples
eeg_inlet = StreamInlet(eeg_streams_info[0],max_buflen=500, max_chunklen=500)
marker_inlet = StreamInlet(marker_streams_info[0])
# %%
# All the possible properties for eeg stream
print('EEG stream info:\n')
print(f'Name: {eeg_inlet.info().name()}')
print(f'Type: {eeg_inlet.info().type()}')
print(f'Channel count: {eeg_inlet.info().channel_count()}')
print(f'Sampling rate: {eeg_inlet.info().nominal_srate()}')
print(f'Channel format: {eeg_inlet.info().channel_format()}')
print(f'Source ID: {eeg_inlet.info().source_id()}')
print(f'Protocol version: {eeg_inlet.info().version()}')
print(f'Stream created at: {eeg_inlet.info().created_at()}')
print(f'Unique ID of the outlet: {eeg_inlet.info().uid()}')
print(f'Session ID: {eeg_inlet.info().session_id()}')
print(f'Host name: {eeg_inlet.info().hostname()}')
print(f'Extended description: {eeg_inlet.info().desc()}')
# print(f'Stream info in XML format:\n{inlet.info().as_xml()}')

# All the possible properties for marker stream
print('\nMarker stream info:\n')
print(f'Name: {marker_inlet.info().name()}')
print(f'Type: {marker_inlet.info().type()}')
print(f'Channel count: {marker_inlet.info().channel_count()}')
print(f'Sampling rate: {marker_inlet.info().nominal_srate()}')
print(f'Channel format: {marker_inlet.info().channel_format()}')
print(f'Source ID: {marker_inlet.info().source_id()}')
print(f'Protocol version: {marker_inlet.info().version()}')
print(f'Stream created at: {marker_inlet.info().created_at()}')
print(f'Unique ID of the outlet: {marker_inlet.info().uid()}')
print(f'Session ID: {marker_inlet.info().session_id()}')
print(f'Host name: {marker_inlet.info().hostname()}')
print(f'Extended description: {marker_inlet.info().desc()}')
# print(f'Stream info in XML format:\n{inlet.info().as_xml()}')

eeg_inlet.open_stream()
marker_inlet.open_stream()

# %%
# Define the StreamInfo object or METADATA
info = StreamInfo(
    name='online_pr_FSM',
    type='Markers',
    channel_count=1,
    nominal_srate=500,
    channel_format='string',
    source_id='test_id',
)

# Create the StreamOutlet object
#chunk_size and max_buffered has default values of chunk_size = 1 and max_buffered = chunk_size. This is what we need for our markers so no need to specify them in StreamOutlet
outlet = StreamOutlet(info,chunk_size=50, max_buffered=50)

# Destroy the StreamInfo object to save space (optional)
info.__del__()
outlet.have_consumers()

# %%

eeg_samples = []
eeg_timestamps = []
marker_samples = []
marker_timestamps = []
corrected_marker_timestamps = []
while(True):
    if eeg_inlet.samples_available():
        eeg_sample, eeg_timestamp = eeg_inlet.pull_sample()
        eeg_samples.append(eeg_sample)
        eeg_timestamps.append(eeg_timestamp)
        if marker_inlet.samples_available():
            marker_sample, marker_timestamp = marker_inlet.pull_sample()
            # Compute latency between timestamps
            latency = eeg_timestamp - marker_timestamp
            # Adjust marker timestamp so it match with eeg timestamps
            corrected_marker_timestamp = marker_timestamp + latency
            marker_samples.append(marker_sample)
            marker_timestamps.append(marker_timestamp)
            corrected_marker_timestamps.append(corrected_marker_timestamp)
            if  marker_sample[0] == "End":
                receive_data_flag = False
                break
                    
np.save('eeg_samples.npy', eeg_samples)
np.save('eeg_timestamps.npy', eeg_timestamps)
np.save('marker_samples.npy', marker_samples)
np.save('marker_timestamps.npy', marker_timestamps)
np.save('corrected_marker_timestamps.npy', corrected_marker_timestamps)
# %%
