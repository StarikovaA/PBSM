# %%
from pylsl import StreamInlet, resolve_byprop
# %%
# Find the stream by its name and type
inlet_name = 'online_pr_FSM'
#inlet_name = 'speller_matrix_markers_online'
info = resolve_byprop('name', inlet_name)

# Create an inlet for the first found stream
inlet = StreamInlet(info[0])

# %%
# All the possible properties
print(f'Name: {inlet.info().name()}')
print(f'Type: {inlet.info().type()}')
print(f'Channel count: {inlet.info().channel_count()}')
print(f'Sampling rate: {inlet.info().nominal_srate()}')
print(f'Channel format: {inlet.info().channel_format()}')
print(f'Source ID: {inlet.info().source_id()}')
print(f'Protocol version: {inlet.info().version()}')
print(f'Stream created at: {inlet.info().created_at()}')
print(f'Unique ID of the outlet: {inlet.info().uid()}')
print(f'Session ID: {inlet.info().session_id()}')
print(f'Host name: {inlet.info().hostname()}')
print(f'Extended description: {inlet.info().desc()}')
# print(f'Stream info in XML format:\n{inlet.info().as_xml()}')

# %%
# Continuously receive and process data
while True:
    # Wait to receive a sample from the outlet
    sample, timestamp = inlet.pull_sample()

    # Check if a valid sample has been received
    if sample is not None:
        # Process the received sample
        marker = sample[0]
        print(f"Received marker: {marker}")

    # Perform any other processing or logic here

    # You can add an exit condition to terminate the loop if needed
# %%
