# %%
import pygame as pg
from pylsl import StreamInlet, resolve_byprop


# %%
# Draw the grid
def draw_grid():
    for i in range(0, 630, 25):
        pg.draw.line(screen, WHITE, (0, i), (630, i), 1)
        pg.draw.line(screen, WHITE, (i, 0), (i, 630), 1)


# Display text on the screen
def message_display(text, text_position_x, text_position_y, textsize):
    largeText = pg.font.SysFont('arial', textsize)
    TextSurf, TextRect = text_objects(text, largeText)
    TextRect.bottomleft = (text_position_x, text_position_y)
    screen.blit(TextSurf, TextRect)
    pg.display.update()


# Draw the initial display
def draw_initial():
    pg.draw.rect(screen, WHITE, (225, 600, width1, height1))
    for indy in [*range(0, 10), *range(12, 25)]:
        pg.draw.rect(screen, GREEN, (7 * 25, 25 * (24 - indy), width1, height1))

    for indxy in range(0, 14):
        pg.draw.rect(screen, GREEN, ((7 + indxy) * 25, 25 * 0, width1, height1))
        pg.draw.rect(screen, GREEN, (20 * 25, 25 * (7 + indxy), width1, height1))

    for indx in range(0, 9):
        pg.draw.rect(screen, GREEN, ((12 + indx) * 25, 25 * 5, width1, height1))
        pg.draw.rect(screen, GREEN, ((12 + indx) * 25, 25 * 7, width1, height1))
        pg.draw.rect(screen, GREEN, ((12 + indx) * 25, 25 * 20, width1, height1))
        pg.draw.rect(screen, GREEN, ((7 + indx) * 25, 25 * 12, width1, height1))
        pg.draw.rect(screen, GREEN, ((7 + indx) * 25, 25 * 15, width1, height1))

    for indy in range(20, 25):
        pg.draw.rect(screen, GREEN, (12 * 25, 25 * (indy), width1, height1))

    pg.draw.rect(screen, GREEN, (12 * 25, 25 * 6, width1, height1))
    pg.draw.rect(screen, GREEN, (15 * 25, 25 * 13, width1, height1))
    pg.draw.rect(screen, GREEN, (15 * 25, 25 * 14, width1, height1))

    for indy in range(0, 4):
        pg.draw.rect(screen, RED, (20 * 25, 25 * (indy + 1), width1, height1))

    draw_grid()
    pg.display.update()


# Display the input instructions
def draw_instructions():
    message_display('UP: W', 650, width1 * 2, 25)
    message_display('LEFT: A ', 650, width1 * 3, 25)
    message_display('RIGHT: D', 650, width1 * 4, 25)
    # message_display('PREV. MOVE: S ', 650, width1 * 5, 25)
    message_display('EXIT: ESC ', 650, width1 * 6, 25)
    pg.display.update()


def leftwall(x, y):
    if (0 * 25 < y < 12 * 25) | (15 * 25 < y < 25 * 25):
        if (x - speed) > (7 * 25):
            finalspeed = speed
        elif (x - speed * 2 / 3) > (7 * 25):
            finalspeed = speed * 2 / 3
        elif (x - speed / 3) > (7 * 25):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    elif 11 * 25 < y < 16 * 25:
        if (x - speed) > (15 * 25):
            finalspeed = speed
        elif (x - speed * 2 / 3) > (15 * 25):
            finalspeed = speed * 2 / 3
        elif (x - speed / 3) > (15 * 25):
            finalspeed = speed / 3
        else:
            finalspeed = 0
    return finalspeed


def rightwall(x, y):
    if 0 * 25 < y < 5 * 25:
        if (x + speed) > (19 * 25):
            finalspeed = speed
            won = 1
            message_display('You Won', 650, width1 * 10, 25)
            message_display('Press SPACE to restart', 650, width1 * 12, 25)
        elif (x + speed) < (20 * 25):
            finalspeed = speed
            won = 0

    elif (4 * 25 < y < 8 * 25) | (19 * 25 < y < 25 * 25):
        won = 0
        if (x + speed) < (12 * 25):
            finalspeed = speed
        elif (x + speed * 2 / 3) < (12 * 25):
            finalspeed = speed * 2 / 3
        elif (x + speed / 3) < (12 * 25):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    elif 7 * 25 < y < 21 * 25:
        won = 0
        if (x + speed) < (20 * 25):
            finalspeed = speed
        elif (x + speed * 2 / 3) < (20 * 25):
            finalspeed = speed * 2 / 3
        elif (x + speed / 3) < (20 * 25):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    return finalspeed, won


def upwall(x, y):
    if (7 * 25 < x < 16 * 25) & (15 * 25 < y < 25 * 25):
        if 15 * 25 < (y - speed):
            finalspeed = speed
        elif 15 * 25 < (y - speed * 2 / 3):
            finalspeed = speed * 2 / 3
        elif 15 * 25 < (y - speed / 3):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    elif (15 * 25 < x < 20 * 25) & (15 * 25 < y < 25 * 25):
        finalspeed = speed

    elif (11 * 25 < x < 20 * 25) & (7 * 25 < y < 16 * 25):
        if 7 * 25 < (y - speed):
            finalspeed = speed
        elif 7 * 25 < (y - speed * 2 / 3):
            finalspeed = speed * 2 / 3
        elif 7 * 25 < (y - speed / 3):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    elif (7 * 25 < x < 20 * 25) & (0 * 25 < y < 9 * 25):
        if 0 * 25 < (y - speed):
            finalspeed = speed
        elif 0 * 25 < (y - speed * 2 / 3):
            finalspeed = speed * 2 / 3
        elif 0 * 25 < (y - speed / 3):
            finalspeed = speed / 3
        else:
            finalspeed = 0

    elif (7 * 25 < x < 12 * 25) & (7 * 25 < y < 12 * 25):
        finalspeed = speed

    return finalspeed


# Move the player up
def move_up(x, y, x_p, y_p):
    pg.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pg.draw.rect(screen, WHITE, (x, y, width1, height1))
    pg.draw.line(screen, GREEN, (x_p + 12.5, y_p + 12.5), (x + 12.5, y + 25), 2)
    pg.draw.line(screen, GREEN, (x + 7.5, y + 30), (x + 12.5, y + 25), 2)
    pg.draw.line(screen, GREEN, (x + 17.5, y + 30), (x + 12.5, y + 25), 2)


# Move the player left
def move_left(x, y, x_p, y_p):
    pg.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pg.draw.rect(screen, WHITE, (x, y, width1, height1))
    pg.draw.line(screen, GREEN, (x_p + 12.5, y_p + 12.5), (x + 25, y + 12.5), 2)
    pg.draw.line(screen, GREEN, (x + 30, y_p + 7.5), (x + 25, y + 12.5), 2)
    pg.draw.line(screen, GREEN, (x + 30, y_p + 17.5), (x + 25, y + 12.5), 2)


# Move the player right
def move_right(x, y, x_p, y_p):
    pg.draw.rect(screen, BLUE, (x_p, y_p, width1, height1))
    pg.draw.rect(screen, WHITE, (x, y, width1, height1))
    pg.draw.line(screen, GREEN, (x_p + 12.5, y_p + 12.5), (x, y + 12.5), 2)
    pg.draw.line(screen, GREEN, (x - 5, y_p + 7.5), (x, y + 12.5), 2)
    pg.draw.line(screen, GREEN, (x - 5, y_p + 17.5), (x, y + 12.5), 2)


# Define text properties
def text_objects(text, font):
    textSurface = font.render(text, True, WHITE)
    return textSurface, textSurface.get_rect()


# Display the previous input key
def draw_input_key():
    key_name = pg.key.name(event.key).upper()
    if key_name == 'LEFT':
        key_name = 'A'
    elif key_name == 'UP':
        key_name = 'W'
    elif key_name == 'RIGHT':
        key_name = 'D'
    # elif key_name == 'DOWN':
    #     key_name = 'S'
    if key_name in ('A', 'W', 'D'):
        message_display(key_name, (num_move - 1) * width1, 700, 20)
        return True


# Show the input characters
def draw_input_lsl(command):
    if command == 'left':
        command = 'A'
        print_message = True
    elif command == 'up':
        command = 'W'
        print_message = True
    elif command == 'right':
        command = 'D'
        print_message = True
    # elif command == 'down':
    #     command = 'S'
    #     print_message = True
    elif command == 'dummy':
        print_message = False
    if print_message:
        if command in ('A', 'W', 'D'):
            message_display(command, (num_move - 1) * width1, 700, 20)


# Initialization
pg.init()

# Define the screen size
size = width, height = 25 * 25 + 300, 25 * 25 + 100
# Set window size
screen = pg.display.set_mode(size)

# Set the window caption
pg.display.set_caption('BCI Maze Game')

# Set the size of the robot
width1, height1 = 25, 25
# Set the speed of the robot
speed = 75
# Set the initial position of the robot
x, y = 225, 600
# Initialize the variables to store previous positions of the robot
x_p, y_p = 225, 600

# Record the last step
last_step = pg.K_f

# Initialize the number of movements
num_move = 1

# Define the keys for controlling the game
list_left = [pg.K_a, pg.K_LEFT]
list_up = [pg.K_w, pg.K_UP]
list_right = [pg.K_d, pg.K_RIGHT]
list_rep = [pg.K_s, pg.K_DOWN]

# Define the colors
WHITE = pg.Color('white')
BLUE = pg.Color('blue')
GREEN = pg.Color('green')
BLACK = pg.Color('black')
RED = pg.Color('red')

# Initialize a variable for the inlet
inlet = None

# Get a StreamInfo object from the available streams
info = resolve_byprop('name', 'ssvep_prediction_markers', timeout=1)

if info:
    # Get the first element of the list
    info = info[0]
    # Create a StreamInlet object
    # max_buflen: Maximum amount of data to buffer in seconds
    inlet = StreamInlet(info, max_buflen=60, max_chunklen=0)

print(inlet)

# Initialize the game loop flag
game_running = True

# Main game loop
while game_running:
    draw_initial()
    draw_instructions()

    for event in pg.event.get():
        # If the window is closed
        if event.type == pg.QUIT:
            pg.display.quit()  # Kill the window
            game_running = False
            break

        # Terminate the game when 40 operations are completed
        elif num_move > 40:
            if num_move == 41:
                message_display('Game Over', 650, width1 * 12, 25)
                message_display('Press SPACE to restart', 650, width1 * 15, 25)
            if event.type == pg.KEYDOWN:  # after one input from the keyboard
                if event.key == pg.K_SPACE:  # if input is space, reset the state
                    num_move = 1
                    x, y = 225, 600  # initial position of the robot
                    x_p, y_p = 225, 600  # previous position of the robot
                    screen.fill(BLACK)
                    draw_initial()
                elif event.key == pg.K_ESCAPE:
                    pg.display.quit()  # Kill the window
                    game_running = False
                    break

        # If there is an LSL stream
        elif inlet is not None:
            # If a key is pressed
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()  # Kill the window
                    game_running = False
                    break

            else:
                # Pull the available sample
                sample, _ = inlet.pull_sample(timeout=0.01)
                if sample is not None:
                    sample = sample[0]
                    sample = sample.split('-')[1]
                    print(sample)
                else:
                    sample = 'dummy'

                draw_input_lsl(sample)

                if sample == 'left':
                    finalspeed = leftwall(x, y)
                    if finalspeed > 0:
                        x_p, y_p = x, y  # record the previous position of the robot
                        x -= finalspeed
                        move_left(x, y, x_p, y_p)
                    next_step = True
                elif sample == 'right':
                    finalspeed, won = rightwall(x, y)
                    if finalspeed > 0:
                        x_p, y_p = x, y  # record the previous position of the robot
                        x += finalspeed
                        move_right(x, y, x_p, y_p)
                    if won > 0:
                        num_move = 50
                    next_step = True
                elif sample == 'up':
                    finalspeed = upwall(x, y)
                    if finalspeed > 0:
                        x_p, y_p = x, y  # record the previous position of the robot
                        y -= finalspeed
                        move_up(x, y, x_p, y_p)
                    next_step = True
                elif sample == 'dummy':
                    next_step = False
                # last_step = event.key
            if next_step:
                num_move += 1  # record the number of operations
                draw_grid()
                print('Move number:', num_move)

        # If the player is controlled with key presses
        else:
            # If a key is pressed
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    pg.display.quit()  # Kill the window
                    game_running = False
                    break

                else:
                    player_moved = draw_input_key()
                    if player_moved:
                        print('Move number:', num_move)
                        num_move += 1  # record the number of operations
                    if event.key in list_rep:
                        event.key = last_step  # repeat the last operation
                    elif event.key in list_left:
                        finalspeed = leftwall(x, y)
                        if finalspeed > 0:
                            x_p, y_p = x, y  # record the previous position of the robot
                            x -= finalspeed
                            move_left(x, y, x_p, y_p)
                    elif event.key in list_right:
                        finalspeed, won = rightwall(x, y)
                        if finalspeed > 0:
                            x_p, y_p = x, y  # record the previous position of the robot
                            x += finalspeed
                            move_right(x, y, x_p, y_p)
                        if won > 0:
                            num_move = 50

                    elif event.key in list_up:
                        finalspeed = upwall(x, y)
                        if finalspeed > 0:
                            x_p, y_p = x, y  # record the previous position of the robot
                            y -= finalspeed
                            move_up(x, y, x_p, y_p)
                    last_step = event.key
                draw_grid()

        # Impose limits
        if x < 0:
            x = 0
        if x > 600:
            x = 600
        if y < 0:
            y = 0
        if y > 600:
            y = 600
        pg.display.update()  # update the window
