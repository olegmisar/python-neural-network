import pygame, sys
import math
from nn import NeuralNetwork
from PIL import Image

nn = NeuralNetwork(784, 200, 10)
nn.load('weights60000_100hidden.json')

pygame.init()
screen = pygame.display.set_mode((560, 560))

is_drawing = False
predict_digit = False
points = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                is_drawing = False
                predict_digit = True
            if event.button == 3:
                points = []

    if is_drawing:
        pos = pygame.mouse.get_pos()
        points.append(pos)

    # Draw
    screen.fill((0, 0, 0))

    for pt in points:
        pygame.draw.circle(screen, (0xFFFFFF), pt, 24)

    pygame.display.flip()

    if predict_digit:
        predict_digit = False
        screenshot = pygame.image.tostring(screen, 'RGB')
        image = Image.frombytes('RGB', screen.get_size(), screenshot).convert('L').resize((28, 28), Image.BICUBIC)
        output = nn.predict(image.getdata())
        result = output.index(max(output))
        print(result)
