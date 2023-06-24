# %%
#%pip install pygame
#If No module named 'pygame', run previous line
import numpy as np
import pygame
import random
import time
from pygame.locals import *
from q_function import show_instruction
from pylsl import StreamInfo, StreamOutlet, local_clock

# %%
# Lab Streaming Layer outlet for markers

# Define the StreamInfo object or METADATA
info = StreamInfo(
    name='speller_matrix_markers_training',
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

#
highlight_number_per_task = np.repeat(["0", "1", "2"], 10)
random.shuffle(highlight_number_per_task)

print(highlight_number_per_task)

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
face = pygame.image.load('face.png')
face_width = face.get_width()
face_height = face.get_height()
 
running = True
last_update_time = pygame.time.get_ticks()
#update_interval = 1280  # literature said so
update_interval = 128  # literature said so

# New variables for game state and iteration count
game_state = 0
scene_state = 0
iteration_count = 0

# CHANGE
digit = 0
instruction_number_index = 0

highlighted_symbols = []  # Keep track of the highlighted symbols

while running:
    if instruction_number_index == len(highlight_number_per_task):
        running = False
    if game_state == 0:
        if (running is not False):            
            if scene_state == 0:
                # Display black screen for 5 seconds
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
                time.sleep(5)
                scene_state = 1
            elif scene_state == 1:
                show_instruction(int(highlight_number_per_task[instruction_number_index]))
                time.sleep(5)
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
            for symbol, position in zip(matrix_symbols, symbol_positions):
                font_size = int(0.3 * min(window_width_px, window_height_px))  # Adjust font size based on window size
                font = pygame.font.Font(None, font_size)
                color = symbol_colors[matrix_symbols.index(symbol)]
                if symbol == highlighted_symbol:
                    #text_color = highlighted_color
                    face_x = position[0] - face_width // 2
                    face_y = position[1] - face_height // 2
                    face_position = (face_x, face_y)
                    window.blit(face, face_position)
                    if symbol == highlight_number_per_task[instruction_number_index]:
                        marker = 'S10'
                        outlet.push_sample([marker], time.time(), pushthrough=True)
                    else:
                        marker = 'S11'
                        outlet.push_sample([marker], time.time(), pushthrough=True)
                else:
                    text_color = color
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
            if iteration_count == 45:
                game_state = 2
                iteration_count = 0

    elif game_state == 2:
        # Display black screen for 2 seconds
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        time.sleep(2)
        game_state = 3
        
    elif game_state == 3:
        instruction_number_index = instruction_number_index + 1
        game_state = 0
pygame.quit()


# %%
