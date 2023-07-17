# %%

#If No module named 'pygame', run previous line
import numpy as np
import pygame
import random
import time
from pygame.locals import *
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop, local_clock

# Lab Streaming Layer outlet for markers

# Define the StreamInfo object or METADATA
info = StreamInfo(
    name='speller_matrix_markers_online',
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

#Create inlet to receive the states from online_pr script
states_inlet_name = 'online_pr_FSM'
states_streams_info = resolve_byprop('name', states_inlet_name)
states_inlet = StreamInlet(states_streams_info[0])

# All the possible properties for marker stream
print('\nMarker stream info:\n')
print(f'Name: {states_inlet.info().name()}')
print(f'Type: {states_inlet.info().type()}')
print(f'Channel count: {states_inlet.info().channel_count()}')
print(f'Sampling rate: {states_inlet.info().nominal_srate()}')
print(f'Channel format: {states_inlet.info().channel_format()}')
print(f'Source ID: {states_inlet.info().source_id()}')
print(f'Protocol version: {states_inlet.info().version()}')
print(f'Stream created at: {states_inlet.info().created_at()}')
print(f'Unique ID of the outlet: {states_inlet.info().uid()}')
print(f'Session ID: {states_inlet.info().session_id()}')
print(f'Host name: {states_inlet.info().hostname()}')
print(f'Extended description: {states_inlet.info().desc()}')
# print(f'Stream info in XML format:\n{inlet.info().as_xml()}')

states_inlet.open_stream()

# Initialize Pygame
pygame.init()

# Set the screen resolution to fullscreen
screen_resolution = pygame.display.Info()
window_width_px = screen_resolution.current_w
window_height_px = screen_resolution.current_h

window = pygame.display.set_mode((window_width_px, window_height_px), pygame.FULLSCREEN)
pygame.display.set_caption("P300")

matrix_symbols = ["0", "1", "2"]
symbol_colors = [
    (150, 150, 150, 255),  # All light gray
    (150, 150, 150, 255),
    (150, 150, 150, 255),
]

number_size = 50

highlighted_color = (255, 255, 255, 255)  # White color for highlighted symbols

# Calculate the gap between symbols based on the window width
#horizontal_gap = window_width_px // (len(matrix_symbols) + 2)
horizontal_gap = window_width_px // 2 - number_size // 2
# Define the horizontal position for the first symbol (0)
start_horizontal_position = horizontal_gap

# Define the vertical position for the symbols (centered vertically)
#vertical_position = window_height_px // 2
vertical_position = window_height_px // 2 - number_size // 2

# Calculate the positions for the symbol

#Horizontal position
symbol_positions = [
    (number_size, vertical_position),            # Position for number 0 (top left corner)
    (horizontal_gap, vertical_position),         # Position for number 1 (center)
    (window_width_px - number_size, vertical_position)  # Posición del número 2 (bottom right corner)
]
font = pygame.font.Font(None, number_size)

##################################################################################
# Main game loop
face = pygame.image.load('face_white.png')
face_width = face.get_width()
face_height = face.get_height()

#calling pic
call = pygame.image.load('call-up.png')
call_w = call.get_width()
call_h = call.get_height()
 
running = True
last_update_time = pygame.time.get_ticks()
update_interval = 12

# New variables for game state and iteration count
game_state = 0
scene_state = 0
iteration_count = 0

# CHANGE
digit = 0
instruction_number_index = 0

#
trials_per_task = 45

#matrix to display selected digits
chosen_number = []

highlighted_symbols = []  # Keep track of the highlighted symbols

while running:
    if game_state == 0:
        if (running is not False):            
            if scene_state == 0:
                # Display black screen for 5 seconds
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
                time.sleep(5)
                scene_state = 1
                marker = 'Init'
                outlet.push_sample([marker], pushthrough=True)
            elif scene_state == 1:
                # Show caption during 5 seconds
                color = (150, 150, 150, 255)  # Light Gray
                text = font.render("Focus on the number you want to select", True, color)
                text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
                window.blit(text, text_rect)
                pygame.display.update()
                time.sleep(5)
                window.fill((0, 0, 0))  # Black
                # Draw the symbols on the window with light gray color
                for symbol, position in zip(matrix_symbols, symbol_positions):
                    #font_size = int(0.18 * min(window_width_px, window_height_px))
                    color = (150, 150, 150, 255)  # Light Gray
                    text = font.render(symbol, True, color)
                    text_rect = text.get_rect(center=position)
                    window.blit(text, text_rect)
                # Update the display
                pygame.display.update()
                marker = 'StartTask'
                outlet.push_sample([marker], pushthrough=True)
                time.sleep(2)
                game_state = 1  # Move to the next game state
    elif game_state == 1:
        # Iterate the main game loop 15 times
        for _ in range(trials_per_task):
            window.fill((0, 0, 0))  # Black
            # Update the symbol colors every 128 milliseconds
            current_time = pygame.time.get_ticks()
            if current_time - last_update_time >= update_interval:
                last_update_time = current_time
                random.shuffle(symbol_colors)  # Randomly shuffle the symbol colors
            
            highlighted_symbol = None  # The symbol to be highlighted
            last_highlighted_symbol = None # Previous highlighted symbol
            
            if len(highlighted_symbols) == len(matrix_symbols):
                # If all symbols have been highlighted, reset the list
                highlighted_symbols = []
            
            # Randomly select a symbol that hasn't been highlighted yet
            while highlighted_symbol == None or highlighted_symbol in highlighted_symbols or highlighted_symbol == last_highlighted_symbol:
                highlighted_symbol = random.choice(matrix_symbols)

            last_highlighted_symbol = None #The while loop above ensures that in case the last symbol to be highlighted during the 
                                           #previous trial was the symbol of interest (the symbol the participant should focus on) is 
                                           #equal to the first symbol to be highlighted in the current trial, then choose another symbol. 
                                           #In this way symbols of interest will not be highlighted consecutively between trials
            
            highlighted_symbols.append(highlighted_symbol)
            
            # Draw the symbols on the window
            for symbol, position in zip(matrix_symbols, symbol_positions):
                color = symbol_colors[matrix_symbols.index(symbol)]
                if symbol == highlighted_symbol:                 
                    face_x = position[0] - face_width // 2
                    face_y = position[1] - face_height // 2
                    face_position = (face_x, face_y)
                    window.blit(face, face_position)
            
                    if symbol == "0":
                        marker = 'S0'
                        outlet.push_sample([marker], pushthrough=True)
                    if symbol == "1":
                        marker = 'S1'
                        outlet.push_sample([marker], pushthrough=True)
                    if symbol == "2":
                        marker = 'S2'
                        outlet.push_sample([marker], pushthrough=True)
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
            time.sleep(0.1)

            # Increment the iteration count
            iteration_count += 1
            if iteration_count == trials_per_task:
                game_state = 2
                iteration_count = 0
        
    elif game_state == 2:
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        time.sleep(2)
        #Lets online_pr known we already send all number trials
        marker = 'End'
        outlet.push_sample([marker], pushthrough=True)
        
        # Display black screen for 2 seconds
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        
        #Wait until online_pr process data
        
        state_sample = "10"#Set a default value before receiving the marker
        state_sample,_ = states_inlet.pull_sample()
        while(state_sample[0] != "0" and state_sample[0] != "1" and state_sample[0] != "2" and state_sample[0] != "4"):
            state_sample,_ = states_inlet.pull_sample()
        digit = state_sample[0]
        print(digit)
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        color = (150, 150, 150, 255)  # Light Gray
        text = font.render("Wait for calibration...", True, color)
        text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
        window.blit(text, text_rect)
        pygame.display.update()
        marker = 'Calibration'
        outlet.push_sample([marker], pushthrough=True) 
        time.sleep(2)
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        if (digit != "4"):
            
            color = (150, 150, 150, 255)  # Light Gray
            text = font.render("Is this the number you selected?: " + digit +" Blink, if Yes.", True, color)
            text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
            window.blit(text, text_rect)
            pygame.display.update()            
            state_sample = "10"#Set a default value before receiving the marker
            state_sample,_ = states_inlet.pull_sample() 
            while(state_sample[0] != "Yes" and state_sample[0] != "No"):
                state_sample,_ = states_inlet.pull_sample()
                print(state_sample[0])
            time.sleep(1)
            if state_sample[0] == "Yes":
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
                chosen_number.append(digit)
                color = (150, 150, 150, 255)  # Light Gray
                text = font.render(f"Now your selected numbers are: " + ''.join(chosen_number), True, color)
                text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
                window.blit(text, text_rect)
                pygame.display.update()
                time.sleep(2)
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
            else:
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
                color = (150, 150, 150, 255)  # Light Gray
                text = font.render("Failed detection. Try again", True, color)
                text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
                window.blit(text, text_rect)
                pygame.display.update()
                time.sleep(1)
                window.fill((0, 0, 0))  # Black
                pygame.display.update()
        else:
            color = (150, 150, 150, 255)  # Light Gray
            text = font.render("Selection not detected, please try again", True, color)
            text_rect = text.get_rect(center=(window_width_px // 2, window_height_px // 2))
            window.blit(text, text_rect)
            pygame.display.update()
            time.sleep(3)
            window.fill((0, 0, 0))  # Black
            pygame.display.update()
        if len(chosen_number) ==3:
            game_state = 4
        else:
            game_state = 3
        
    elif game_state == 3:
        instruction_number_index = instruction_number_index + 1
        game_state = 0
        marker = 'NotFinish'
        outlet.push_sample([marker], pushthrough=True)
    elif game_state == 4:
        call_position = call.get_rect(center=(window_width_px // 2, window_height_px // 2))
        window.blit(call, call_position)
        pygame.display.update()
        time.sleep(3)
        marker = 'Finish'
        outlet.push_sample([marker], pushthrough=True)
        break


pygame.quit()


# %%
