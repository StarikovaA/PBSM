# %%
import numpy as np
import mne
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
import matplotlib
import matplotlib.pyplot as plt
from enum import Enum
import joblib
from eye_bklinking_detection.eeg_eye_blink_detection import eye_blink_detection

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

# Define states
class States(Enum):
    IDLE = 0
    ACQUISITION = 1
    PROCESSING = 2
    FEATURE_EXTRACTION = 3
    CLASSIFICATION = 4
    WAIT_BLINK_CONFIRMATION = 5
    END = 6
    
def find_nearest_indices(arr1, arr2):
    indices = []
    for value in arr1:
        abs_diff = np.abs(arr2 - value)
        nearest_index = np.argmin(abs_diff) #The minimum absolute value between the arrays is calculated. For example if arr2 has elements 454 and 456 and value is 455, 
                                            #the subtraction abs(454-455) = 1 and abs(456-455) = 1, how then select only one of them? np.argmin() returns only the index of 
                                            #the first minimum found in case there are several minimum values ​​in the array.
        indices.append(nearest_index)
    return indices
    
sampling_rate = eeg_inlet.info().nominal_srate()

#Best features
file_path = 'trained_model/best_features_Familiar_Face_idx.txt'# File path where the best features index was saved
best_features_idx = np.loadtxt(file_path, dtype=int)# Load the best features index from the file

#Trained Classifier
file_path = 'trained_model/lda_model_Familiar_Face.pkl'# File path where model was saved
loaded_model = joblib.load(file_path)# Load the model

STATE = States.IDLE
keyboard = '0'
eeg_samples = []
eeg_timestamps = []
marker_samples = []
marker_timestamps = []
corrected_marker_timestamps = []

while(True):
    if STATE == States.IDLE:
        marker_sample, _ = marker_inlet.pull_sample()
        if (marker_sample[0] == "Init"):
            eeg_inlet.flush()
            marker_inlet.flush()
            STATE = States.ACQUISITION
    if STATE == States.ACQUISITION:
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
                    STATE = States.PROCESSING
    if STATE == States.PROCESSING:
        eeg_samples = np.transpose(np.array(eeg_samples))
        eeg_timestamps = np.array(eeg_timestamps)
        marker_samples = np.array(marker_samples)
        corrected_marker_timestamps = np.array(corrected_marker_timestamps)
        low_bandpass_fr = 2
        high_bandpass_fr = 16
        eeg_samples_filter = mne.filter.filter_data(eeg_samples.astype(np.float64),sampling_rate,low_bandpass_fr,high_bandpass_fr)
        # To do the epoching we will first split the marker_sample array according to its 
        # marker, S0, S1 or S2. By finding the indices of each one, we can use them to 
        # index their corresponding timestamps. That is, obtain the times in which each 
        # marker occurred. Having this information, we only have to search for these times 
        # within the eeg_samples timestamp, obtain their indices and then use them 
        # to finally index the eeg_samples 
        markers_S0_idx = np.where(marker_samples == "S0")[0]
        markers_S1_idx = np.where(marker_samples == "S1")[0]
        markers_S2_idx = np.where(marker_samples == "S2")[0]
        
        timestamps_S0 = np.take(marker_timestamps, markers_S0_idx)
        timestamps_S1 = np.take(marker_timestamps, markers_S1_idx)
        timestamps_S2 = np.take(marker_timestamps, markers_S2_idx)

        eeg_timestamps_marker_S0 = find_nearest_indices(timestamps_S0,eeg_timestamps)
        eeg_timestamps_marker_S1 = find_nearest_indices(timestamps_S1,eeg_timestamps)
        eeg_timestamps_marker_S2 = find_nearest_indices(timestamps_S2,eeg_timestamps)


        game_trials = len(eeg_timestamps_marker_S0)

        tmin = 0
        tmax = 1
        tmin_samples = np.abs(int(tmin*int(sampling_rate)))
        tmax_samples = np.abs(int(tmax*int(sampling_rate)))

        epochs_S0 = []
        epochs_S1 = []
        epochs_S2 = []
        for game_trial in range(game_trials):
            epochs_S0.append(eeg_samples_filter[:,eeg_timestamps_marker_S0[game_trial]:eeg_timestamps_marker_S0[game_trial] + tmax_samples])
            epochs_S1.append(eeg_samples_filter[:,eeg_timestamps_marker_S1[game_trial]:eeg_timestamps_marker_S1[game_trial] + tmax_samples])
            epochs_S2.append(eeg_samples_filter[:,eeg_timestamps_marker_S2[game_trial]:eeg_timestamps_marker_S2[game_trial] + tmax_samples])

        epochs_S0 = np.array(epochs_S0)
        epochs_S1 = np.array(epochs_S1)
        epochs_S2 = np.array(epochs_S2)
        STATE = States.FEATURE_EXTRACTION
    if STATE == States.FEATURE_EXTRACTION:
        avg_marker_S0 = np.mean(epochs_S0, axis = 0)
        avg_marker_S1 = np.mean(epochs_S1, axis = 0)
        avg_marker_S2 = np.mean(epochs_S2, axis = 0)

        #Select a window to look for maximum and get the indexes
        time = np.arange(tmin,tmax,1/int(sampling_rate))
        lower_time_window = 0.250
        upper_time_window = 0.350
        lower_idx = np.where(time >= lower_time_window)[0][0]
        upper_idx = np.where(time <= upper_time_window)[0][-1]
        
        #plt.plot(time,avg_marker_S0[21,:],color='lightgreen')
        #plt.plot(time,avg_marker_S1[21,:],color='red')
        #plt.plot(time,avg_marker_S2[21,:],color='black')
        #legend_labels = ['0', '1', '2']
        #plt.legend(legend_labels)
        #plt.show()
        
        #Get the maximum amplitude for each average non event trial per task in the given range and store it in the feature vector
        feature_1_S0 = np.max(avg_marker_S0[:, lower_idx:upper_idx], axis=1)
        feature_1_S1 = np.max(avg_marker_S1[:, lower_idx:upper_idx], axis=1)
        feature_1_S2 = np.max(avg_marker_S2[:, lower_idx:upper_idx], axis=1)
        STATE = States.CLASSIFICATION
    if STATE == States.CLASSIFICATION:
        
        # From each event feature vector extract the best features
        best_feature_1_S0 = feature_1_S0[best_features_idx]
        best_feature_1_S1 = feature_1_S1[best_features_idx]
        best_feature_1_S2 = feature_1_S2[best_features_idx]
        
        #Concatenate best features into a single feature vector
        feature_vector = [best_feature_1_S0, best_feature_1_S1, best_feature_1_S2]
        
        #Predict the probability that each input to the classifier (feature row) belongs to each of the two classes (event or non event)
        prediction_prob = loaded_model.predict_proba(feature_vector)
        #Predcit the labels
        prediction = loaded_model.predict(feature_vector)
                
        #Print just for visualization and debugging
        print(prediction[0])
        print(prediction[1])
        print(prediction[2])
        
        
        all_non_event = np.where(prediction == -1)[0] #Detection of event not detected (cases in which there is no P300 in any event (all true negatives) or 
                                                      #that the classifier failed to correctly classify at least one event (false negatives))
        if(len(all_non_event) == 3):
            digit = 4 #This value will tell the experiment that no input to the classifier was classified as an event.
        else:
            #Look for the indices (0,1 or 2) that are predicted to belong to the event class
            prediction_idx = np.where(prediction == 1)[0]
            #Each prediction probability has two values, the first specifies the prediction probability of belonging 
            # to the non envet class, the second specifies the prediction probability of belonging to the event class.
            
            # 1. Obtain the prediction probability of belonging to the event class only for those entries to the classifier 
            # (features row) that have a label that classifies them as event, that is, prediction_prob[prediction_idx][:,1].
            
            # 2. From the above, get the index of the input of the classifier that has the highest prediction probability 
            # (np.argmax(prediction_prob[prediction_idx][:,1])) and use it to define the final prediction of the classifier (digit = prediction_idx[prediction_max_prob_idx])
            prediction_max_prob_idx = np.argmax(prediction_prob[prediction_idx][:,1])
            digit = prediction_idx[prediction_max_prob_idx]
        
        marker = str(digit)
        outlet.push_sample([marker], pushthrough=True)
        if digit == 4:#If there is no event detection jump to END state so the program can reset variables and collect data again
            STATE = States.END
        else:
            STATE = States.WAIT_BLINK_CONFIRMATION #If there is at least one event detection wait for blink confirmation from the user
    if STATE == States.WAIT_BLINK_CONFIRMATION:
        marker_sample = "10"#Set a default value before receiving the marker
        marker_sample,_ = marker_inlet.pull_sample()
        while(marker_sample[0] != "Calibration"):
            marker_sample,_ = marker_inlet.pull_sample()
        #Blinking function
        blink_flag = False
        blink_flag = eye_blink_detection(10)
        print(blink_flag)
        
        #Send yes or no depending on blink detection
        marker = "S0"#Set a default value before receiving the marker
        if (blink_flag):
            marker = "Yes"
            outlet.push_sample([marker], pushthrough=True)
        else:
            marker = "No"
            outlet.push_sample([marker], pushthrough=True)
            
        STATE = States.END
    if STATE == States.END:
        marker_sample = "10"
        marker_sample,_ = marker_inlet.pull_sample()
        while(marker_sample[0] != "Finish" and marker_sample[0] != "NotFinish"):
            marker_sample,_ = marker_inlet.pull_sample()
        if marker_sample[0] == "Finish":
            break
        else:
            eeg_samples = []
            eeg_timestamps = []
            marker_samples = []
            marker_timestamps = []
            corrected_marker_timestamps = []
            eeg_inlet.flush()
            marker_inlet.flush()
            STATE = States.ACQUISITION
