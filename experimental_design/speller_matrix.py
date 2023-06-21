import pygame
import random
import time
from pygame.locals import *
from q_function import ask_question

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
    (255, 255, 255, 255),  # White (0)
    (64, 64, 64, 255),
    (64, 64, 64, 255),
]

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
update_interval = 128  # literature said so

# New variables for game state and iteration count
game_state = 0
scene_state = 0
iteration_count = 0

while running:
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
        for _ in range(10):
            window.fill((0, 0, 0))  # Black
            # Update the symbol colors every 128 milliseconds
            current_time = pygame.time.get_ticks()
            if current_time - last_update_time >= update_interval:
                last_update_time = current_time
                random.shuffle(symbol_colors)  # Randomly shuffle the symbol colors
            
            # Draw the symbols on the window
            for symbol, position in zip(matrix_symbols, symbol_positions):
                font_size = int(0.3 * min(window_width_px, window_height_px))  # Adjust font size based on window size
                font = pygame.font.Font(None, font_size)
                color = symbol_colors[matrix_symbols.index(symbol)]
                text = font.render(symbol, True, color) 
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
            print(iteration_count)
            if iteration_count == 10:
                game_state = 2
                iteration_count = 0

    elif game_state == 2:
        # Display black screen for 2 seconds
        window.fill((0, 0, 0))  # Black
        pygame.display.update()
        time.sleep(2)
        game_state = 3
        
    elif game_state == 3:
        ask_question(0)
        # here could be the part with the eye blinking
        game_state = 0
        


pygame.quit()
