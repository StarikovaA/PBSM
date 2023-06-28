# %%
"""
Real-time detection of eye blinks from a single EEG channel. The detection can
be used as control signals to different peripherals, like ESP32 in this case.

Author:
    Karahan Yilmazer

Email:
    yilmazerkarahan@gmail.com
"""

#! %matplotlib qt

import sys

import socket
from time import time

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec, SubplotSpec
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_stream
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm



# %%
class BlitManager:
    # https://matplotlib.org/stable/tutorials/advanced/blitting.html#sphx-glr-tutorials-advanced-blitting-py
    def __init__(self, fig, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = fig.canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = self.canvas.mpl_connect('draw_event', self.on_draw)
        self.cid = self.canvas.mpl_connect('close_event', self.on_close)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def on_close(self, event):
        global figure_closed
        figure_closed = True

    def add_artist(self, arts):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if type(arts) != list:
            arts = [arts]
        for art in arts:
            if art.figure != self.canvas.figure:
                raise RuntimeError
            art.set_animated(True)
            self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            #print(np.shape(self))
            self._draw_animated()
            #print('test after animated')
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


def receive_eeg_samples(
    inlet, samples_buffer, timestamps_buffer, buffer_size=5000, chunk_size=100
):
    """
    Receives new EEG samples and timestamps from the LSL input and stores them in the
    buffer variables
    :param samples_buffer: list of samples(each itself a list of values)
    :param timestamps_buffer: list of time-stamps
    :return: updated samples_buffer and timestamps_buffer with the most recent 150
    samples
    """

    # Pull a chunk of maximum chunk_size samples
    chunk, timestamps = inlet.pull_chunk(max_samples=chunk_size)

    # If there are no new samples
    if chunk == []:
        # Return the unchanged buffer
        return samples_buffer, timestamps_buffer, 0
    # If there are new samples
    else:
        # Get the number of newly fetched samples
        n_new_samples = len(chunk)

        # Convert the chunk into a np.array and transpose it
        samples = [sample[:24] for sample in chunk]

        # Extend the buffer with the data
        samples_buffer.extend(samples)
        # Extend the buffer with time stamps
        timestamps_buffer.extend(timestamps)

        # Get the last buffer_size samples and time stamps from the buffer
        data_from_buffer = samples_buffer[-buffer_size:]
        timestamps_from_buffer = timestamps_buffer[-buffer_size:]

        return data_from_buffer, timestamps_from_buffer, n_new_samples


def new_subplot(fig, bm, title=None, color='navy', lw=2):
    # Create a new subplot
    tmp_ax = fig.add_subplot()
    tmp_ax.set_title(title)

    # Add artists to the BlitManager to be updated later
    # Real-time data
    bm.add_artist(tmp_ax.plot([], [], color=color, lw=lw))
    # Detection threshold
    bm.add_artist(tmp_ax.axhline(0, lw=2, ls='dashed', color='gray'))
    # Detected peaks
    bm.add_artist(tmp_ax.plot([], [], color='tomato', marker='x', lw=2))

    # Baseline period
    tmp_ax.axvspan(
        buffer_size + base_begin, buffer_size + base_end, alpha=0.1, color='green'
    )
    # Detection window
    tmp_ax.axvspan(
        buffer_size + activity_begin, buffer_size + activity_end, alpha=0.1, color='red'
    )
    # x-axis
    tmp_ax.axhline(0, color='grey')

    tmp_ax.set_xlim([x_min, x_max])
    tmp_ax.set_ylim([y_min, y_max])

    # Get all the axes from the figure
    axes = fig.axes
    # Get the number of axes
    n_axes = len(axes)
    # Iterate over the axes
    for i, ax in enumerate(axes):
        # Define the position of all subplots using the number of total subplots
        ax.set_subplotspec(SubplotSpec(GridSpec(n_axes, 1), i))

        # Set the labels
        ax.set_ylabel('Amp.')
        if i == n_axes - 1:
            ax.set_xlabel('Sample')
        else:
            ax.set_xlabel('')

    # Set a tight layout for the titles to be clearly readable
    fig.tight_layout()

    return fig


# %%
if __name__ == '__main__':
    print("Looking for an EEG stream")
    streams = resolve_stream()
    print("Found one")
    inlet = StreamInlet(streams[0])
    print("Connected to the inlet")

    # Supress MNE info messages and only show warnings
    mne.set_log_level(verbose='WARNING')

    # Set the channel names
    # ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']  # UHB
    ch_names = [
        'Fp1',
        'Fp2',
        'Fz',
        'F7',
        'F8',
        'FC1',
        'FC2',
        'Cz',
        'C3',
        'C4',
        'T7',
        'T8',
        'CPz',
        'CP1',
        'CP2',
        'CP5',
        'CP6',
        'M1',
        'M2',
        'Pz',
        'P3',
        'P4',
        'O1',
        'O2',
    ]  # Smarting

    # Sampling rate of the Unicorn Hybrid Black
    sfreq = 250  # Hz

    # Define the frequencies for the notch filter
    uhb_notch_freqs = np.arange(50, 101, 50)

    # Define the band-pass cutoff frequencies
    flow, fhigh = 1, 30

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names, sfreq, 'eeg')
    # Add the system name to the info
    # info['description'] = 'Unicorn Hybrid Black'
    info['description'] = 'Smarting 24'

    # Create the outlet stream information
    info_number = StreamInfo(
        name='detected_number_markers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id='detected_number_markers',
    )
    info_eeg = StreamInfo(
        name='filtered_data_stream',
        type='eeg',
        channel_count=24,
        channel_format='float32',
        source_id='smarting',
    )

    # Create the outlet to send data
    stream_number = StreamOutlet(info_number)
    stream_eeg = StreamOutlet(info_eeg)

    # Initialize the marker list
    marker_list = []

    # Initalize the buffers
    samples_buffer = []
    timestamps_buffer = []

    # Set the buffer sizes
    buffer_size = 5000
    chunk_size = 500


    # Variables for the calibration
    pbar = tqdm(desc='Calibration', total=buffer_size)
    pbar_closed = False
    old_val = 0

    ####################################################################################
    # TODO: Define all necessary variables

    # Define the length of the moving average filter (arbitrary)
    N = 20
    ####################################################################################

    # ==================================================================================
    # PLOTTING
    # ==================================================================================

    # Set the axis limits
    y_min = -500
    y_max = 500
    x_min = 0
    x_max = buffer_size

    # Define the subplot titles
    title1 = f'Band-Pass Filtered Channel {ch}'
    title2 = f'Moving Average Smoothened Channel {ch} (N={N})'

    # Create an empty figure
    fig = plt.figure()
    bm = BlitManager(fig)
    new_subplot(fig, bm, title=title1)
    new_subplot(fig, bm, title=title2)

    # Define what will happen when the figure is closed
    # fig.canvas.mpl_connect('close_event', on_close)
    # Variable to keep track of the closing of the matplotlib figure
    figure_closed = False

    plt.show(block=False)
    plt.pause(0.1)

    # ==================================================================================
    # Single eye blink detection
    # ==================================================================================
    # Enter the endless loop
    while True:
        # If the matplotlib figure is closed
        if figure_closed:
            # Break out of the loop
            break

        # Get the EEG samples, time stamps and the index that tells from which
        # point on the new samples have been appended to the buffer
        samples_buffer, timestamps_buffer, n_new_samples = receive_eeg_samples(
            inlet,
            samples_buffer,
            timestamps_buffer,
            buffer_size=buffer_size,
            chunk_size=chunk_size,
        )

        # Processing
        # ==============================================================================
        # Check if the calibration is over
        if pbar_closed:
            ############################################################################
            # TODO: Band-pass filter the data
            # Tip: filter_data from MNE
            sfreq = 250 #from lsl_inlet.py
            l_freq = 1
            h_freq = 40 #or 40
            filt_data = mne.filter.filter_data(np.array(samples_buffer).T, sfreq, l_freq, h_freq)
            #filt_data = np.array(samples_buffer).T
            ############################################################################

            ############################################################################
            # TODO: Smoothen the filtered signal with a moving average filter
            # Tip: uniform_filter1d
            moving_average = np.array(uniform_filter1d(filt_data,N)).T
            #print(np.array(moving_average).shape)
            ############################################################################

            #insert the code to extract a time window 500ms after receiving a marker (Sf and St)



            # Plotting
            # ==========================================================================
            # If there are new samples in the buffer
            if n_new_samples > 0:
                # Update the artists (lines) in the subplots
                x_range = range(len(samples_buffer))
                #print('shape filt_data')
                #print(np.shape(filt_data))
                bm._artists[0].set_data(x_range, np.array(filt_data).T[:,idx])
                bm._artists[1].set_data(x_range, thr)
                #print(thr, refractory_period)
                #print('shape moving_average', np.shape(moving_average))
                bm._artists[3].set_data(x_range, moving_average[:,idx])
                bm._artists[4].set_data(x_range, thr)
                bm.update()
                #print('-----------------------')


        # Progress bar
        # ==============================================================================
        else:
            # Get the current number of samples
            len_buffer = len(timestamps_buffer)
            # Calculate the progress update value
            update_val = len_buffer - old_val
            # Store the current number of samples for the next iteration
            old_val = len_buffer
            # Update the progress bar
            pbar.update(update_val)
            # If the progress bar is full
            if len_buffer == buffer_size:
                # Close the progress bar
                pbar.close()
                # Set the flag to get out of the loop
                pbar_closed = True
# %%
