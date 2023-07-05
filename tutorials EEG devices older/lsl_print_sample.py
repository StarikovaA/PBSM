# %%
from pylsl import StreamInlet, resolve_stream

# %%
# Define the LSL stream name
stream_name = 'Liesl-Mock-EEG'
# stream_name = 'Unicorn'

# Connect to the stream
stream = resolve_stream('name', stream_name)
inlet = StreamInlet(stream[0])

# Enter an endless loop
while True:
    # Pull a sample from the inlet
    sample, timestamp = inlet.pull_sample()

    # Print the pulled sample and its timestamp
    print(f'{timestamp}: {sample}\n')
# %%
