# %%
#%pip install pygame
#If No module named 'pygame', run previous line
import numpy as np
import pygame
import random
import time
from pygame.locals import *
from q_function import ask_question
from pylsl import StreamInfo, StreamOutlet, local_clock
from pylsl import StreamInlet, resolve_byprop
# %%
# Find the stream by its name and type
inlet_name = 'eye_blink_detection_markers'
#inlet_name = 'speller_matrix_markers_online'
info = resolve_byprop('name', inlet_name)

# Create an inlet for the first found stream
inlet = StreamInlet(info[0])

# %%
# Lab Streaming Layer outlet for markers

# Define the StreamInfo object or METADATA
info = StreamInfo(
    name='speller_matrix_markers_online',
    type='Markers',
    channel_count=1,
    nominal_srate=0,
    channel_format='string',
    source_id='test_id',
)

# Create the StreamOutlet object
#chunk_size and max_buffered has default values of chunk_size = 1 and max_buffered = chunk_size. This is what we need for our markers so no need to specify them in StreamOutlet
outlet = StreamOutlet(info)

# Destroy the StreamInfo object to save space (optional)
info.__del__()
outlet.have_consumers()
#%%



highlight_number_per_task = np.repeat(["0", "1", "2"], 10)
random.shuffle(highlight_number_per_task)

print(highlight_number_per_task)

#%%
digit_set  = []
# %%
# Initialize Pygame
pygame.init()

desired_width_cm = 13.32
desired_height_cm = 9.98

# Set the screen resolution in pixels
screen_resolution = pygame.display.Info()
screen_width_px = screen_resolution.current_w
screen_height_px = screen_resolution.current_h

# Calculate the scaling factor for converting centimeters to pixels
cm_to_px_scale = min(screen_width_px / desired_width_cm, screen_height_px / desired_height_cm)

# Calculate the window dimensions in pixels
window_width_px = int(desired_width_cm * cm_to_px_scale)
window_height_px = int(desired_height_cm * cm_to_px_scale)

window = pygame.display.set_mode((window_width_px, window_height_px))
pygame.display.set_caption("P300")

matrix_symbols = ["0", "1", "2"]
symbol_colors = [
    (64, 64, 64, 255),  # All light gray
    (64, 64, 64, 255),
    (64, 64, 64, 255),
]

highlighted_color = (255, 255, 255, 255)  # White color for highlighted symbols

horizontal_gap = window_width_px // (len(matrix_symbols) + 1)
vertical_position = window_height_px // 2
start_horizontal_position = horizontal_gap

# Define the positions for the symbols
symbol_positions = []
for i in range(len(matrix_symbols)):
    # Calculate the horizontal position for the current symbol
    horizontal_position = start_horizontal_position + i * horizontal_gap
    # Add the position to the symbol_positions list
    symbol_positions.append((horizontal_position, vertical_position))

##################################################################################
# Main game loop

running = True
last_update_time = pygame.time.get_ticks()
#update_interval = 1280  # literature said so
update_interval = 128  # literature said so

# New variables for game state and iteration count
game_state = 0 #just for testing, otherwise it should be 0
scene_state = 0
iteration_count = 0
counter_task = 0
counter_0 = 0
counter_1 = 0
counter_2 = 0

# CHANGE
digit = 0
instruction_number_index = 0

highlighted_symbols = []  # Keep track of the highlighted symbols

while running:
    if instruction_number_index == len(highlight_number_per_task):
        running = False
    if game_state == 0:
        if scene_state == 0:
            # Display black screen for 5 seconds
            window.fill((0, 0, 0))  # Black
            pygame.display.update()
            time.sleep(5)
            scene_state = 1
        elif scene_state == 1:
            # Display the scene with light gray numbers for 1s
            window.fill((0, 0, 0))  # Black
            # Draw the symbols on the window with light gray color
            for symbol, position in zip(matrix_symbols, symbol_positions):
                font_size = int(0.3 * min(window_width_px, window_height_px))
                font = pygame.font.Font(None, font_size)
                color = (64, 64, 64, 255)  # Light Gray
                text = font.render(symbol, True, color)
                text_rect = text.get_rect(center=position)
                window.blit(text, text_rect)
            # Update the display
            pygame.display.update()
            time.sleep(1)
            game_state = 1  # Move to the next game state
    elif game_state == 1:
        # Iterate the main game loop 15 times
        for _ in range(45):
            window.fill((0, 0, 0))  # Black
            # Update the symbol colors every 128 milliseconds
            current_time = pygame.time.get_ticks()
            if current_time - last_update_time >= update_interval:
                last_update_time = current_time
                random.shuffle(symbol_colors)  # Randomly shuffle the symbol colors
            
            highlighted_symbol = None  # The symbol to be highlighted
            
            if len(highlighted_symbols) == len(matrix_symbols):
                # If all symbols have been highlighted, reset the list
                highlighted_symbols = []
            
            # Randomly select a symbol that hasn't been highlighted yet
            while highlighted_symbol is None or highlighted_symbol in highlighted_symbols:
                highlighted_symbol = random.choice(matrix_symbols)
            
            highlighted_symbols.append(highlighted_symbol)
            
            # Draw the symbols on the window
            marker_sent = False
            for symbol, position in zip(matrix_symbols, symbol_positions):
                font_size = int(0.3 * min(window_width_px, window_height_px))  # Adjust font size based on window size
                font = pygame.font.Font(None, font_size)
                color = symbol_colors[matrix_symbols.index(symbol)]
                text_color = highlighted_color if symbol == highlighted_symbol else color
                #print(f"!!!!!! MARKER IS SENT HERE {highlighted_symbol}")
                if (highlighted_symbol == "0"):
                    if not marker_sent:
                        marker = 'S0'
                        outlet.push_sample([marker], time.time(), pushthrough=True)
                    marker_sent = True
                    '''
                    counter_0 = counter_0 + 1
                    s0 = True
                    s1 = False
                    s2 = False
                    '''
                elif (highlighted_symbol == "1"):
                    if not marker_sent:
                        marker = 'S1'
                        outlet.push_sample([marker], time.time(), pushthrough=True)
                    marker_sent = True
                    '''
                    counter_1 = counter_1 + 1
                    s0 = False
                    s1 = True
                    s2 = False
                    '''
                elif (highlighted_symbol == "2"):
                    if not marker_sent:
                        marker = 'S2'
                        outlet.push_sample([marker], time.time(), pushthrough=True)
                    marker_sent = True
                    '''
                    counter_2 = counter_2 + 1
                    s0 = False
                    s1 = False
                    s2 = True
                    '''
                #print(s0,s1,s2)
                text = font.render(symbol, True, text_color) 
                text_rect = text.get_rect(center=position)
                window.blit(text, text_rect)

            # Update the display
            pygame.display.update()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            pygame.time.wait(update_interval)

            # Increment the iteration count
            iteration_count += 1
            #print(f"Iteration: {iteration_count}, Highlighted Symbol: {highlighted_symbol}")
            if (highlighted_symbol == "0"):
                counter_0 = counter_0 + 1

            elif (highlighted_symbol == "1"):
                counter_1 = counter_1 + 1
            elif (highlighted_symbol == "2"):
                counter_2 = counter_2 + 1
            if iteration_count == 45:
                game_state = 2
                iteration_count = 0

    elif game_state == 2:
        # Display black screen for 2 seconds
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        time.sleep(2)
        game_state = 3	
        counter_task += 1
        #print(f'Task number: {counter_task}, 0: {counter_0}, 1: {counter_1}, 2: {counter_2}')
        
    elif game_state == 3:
        # digit is the value from the classifier	
        ask_question(digit)	
        time.sleep(1)	
        if not marker_sent:
            marker = 'Sbs' #Start of the eye_blinking detection
            outlet.push_sample([marker], time.time(), pushthrough=True)
            marker_sent = True
        # time.sleep(3)
        start_time = time.time()
        time_blink = 2
        while elapsed_time <= time_blink:
            sample, timestamp = inlet.pull_sample()
            if sample is not None:
                marker = 'Sbe' #End of the eye_blinking detection
                outlet.push_sample([marker], time.time(), pushthrough=True)
                digit_set = digit_set.append(digit)  


                instruction_number_index = instruction_number_index + 1
                game_state = 0
                break
            end_time = time.time()
            elapsed_time = end_time - start_time
pygame.quit()


# %%
