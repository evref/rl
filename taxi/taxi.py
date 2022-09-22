import pygame
import sys
import os

from pygame.locals import *

# Initialization
pygame.init()
clock = pygame.time.Clock()
scrn_sz = (840, 840)
screen = pygame.display.set_mode(scrn_sz)

# Load and set variables
pygame.display.set_caption('Taxi scenario')

# Initialize game
def load_env(fname):
    data = []
    with open(fname) as f:
        i = 0
        for val in f.readlines():
            data.append([])
            for c in val:
                if c == '\n':
                    break 
                data[i].append(c)
            i += 1

    return data

# Print grid function
def print_grid(grid):
    for line in grid:
        s = ""
        for c in line:
            s += c
        print(s)

grid = load_env('map')
print_grid(grid)

while True:
    clock.tick(60)
    screen.fill((0,0,0));
    x, y = pygame.mouse.get_pos()

    for event in pygame.event.get():
        # Check key presses
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                sys.exit()
        # Check quit state
        if event.type == pygame.QUIT:
            sys.exit()

    pygame.display.update()
