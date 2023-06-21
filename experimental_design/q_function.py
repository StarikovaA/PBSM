import pygame

#digit = output from classifier

def ask_question (digit):
    question = False #initialize
# the loop for defining the text of the question
    if digit == 0:
        question = 'Did you choose 0? Blink if YES'
    elif digit == 1:
        question = 'Did you choose 1? Blink if YES'
    elif digit == 2:
        question = 'Did you choose 2? Blink if YES'
    #Settin the window with the question

    #setting palette
    txt_color = (132,112,255) #lightslateblue
    bg = (220,220,220) #gray

    #desired size of the window
    qwindow_width_cm = 6
    qwindow_height_cm = 3

    # Calculate the window dimensions in pixels
    screen_resolution = pygame.display.Info() # Set the screen resolution in pixels
    screen_width_px = screen_resolution.current_w
    screen_height_px = screen_resolution.current_h
    cm_to_px_scale = min(screen_width_px / qwindow_width_cm, screen_height_px / qwindow_height_cm)
    window_width_px = int(qwindow_width_cm * cm_to_px_scale)
    window_height_px = int(qwindow_height_cm * cm_to_px_scale)

    # Set the font
    fnt = pygame.font.SysFont("Arial",int(0.5 * min(window_width_px, window_height_px)))

    # render the text
    q_text = fnt.render(question,True, txt_color)

    # launch the window
    qwindow = pygame.display.set_mode((window_width_px, window_height_px))
    qwindow.fill(bg)
    qwindow.blit(q_text,(window_width_px//2 - q_text.get_width() // 2, window_height_px*0.5 - q_text.get_height() // 2))
    pygame.display.flip()
