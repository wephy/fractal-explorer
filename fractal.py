import pygame
from pygame.locals import *

import argparse
import sys

import plot_fractal

parser = argparse.ArgumentParser(description='Fractal Explorer')
parser.add_argument(
    '-f',
    type=str,
    default='mandelbrot',
    dest='FRACTAL',
    help=f'fractal set to generate from {plot_fractal.fractal_sets}'
    )
parser.add_argument(
    '-i',
    type=int,
    default=1500,
    dest='ITERATIONS',
    help=f'maximum iterations'
    )
parser.add_argument(
    '-c',
    default=(-2.5, 1.5, -2.0, 2.0),
    nargs='+',
    dest='COORDS',
    help=f'starting coordinates xmin, xmax, ymin, ymax separated by spaces.\
           e.g. "py fractal.py -c -2.5 1.5 -2.0 2.0"'
    )
parser.add_argument(
    '-r',
    type=int,
    default=720,
    dest='RESOLUTION',
    help=f'resolution of window'
    )

print(parser.format_help())
if sys.argv[0] == 'fractal.py':
    args = parser.parse_args()

    FRACTAL = args.FRACTAL
    ITERATIONS = args.ITERATIONS
    COORDS = tuple(map(float, args.COORDS))
    RESOLUTION = args.RESOLUTION
else:
    FRACTAL = 'mandelbrot'
    ITERATIONS = 1500
    COORDS = (-2.5, 1.5, -2.0, 2.0)
    RESOLUTION = 720


def main(
    fractal=FRACTAL,
    iterations=ITERATIONS,
    resolution=RESOLUTION,
    coords=COORDS
         ):

    # Create coord variables
    min_x, max_x, min_y, max_y = coords
    size = max_x - min_x

    # Render first image (Executed first as a check)
    plot_fractal.render_image(
        coords=coords,
        iterations=iterations,
        resolution=resolution,
        fractal=fractal,
        savename="./render/fractal.png")

    # Pygame setup
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption('Fractal Plot')
    screen = pygame.display.set_mode((resolution, resolution))
    pygame.display.set_caption(f'Fractal plot: {fractal}')

    # Load and show first image
    img = pygame.image.load("./render/fractal.png")
    screen.blit(img, (0, 0))

    # Render zoomed image on mouse button
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos_x, pos_y = pygame.mouse.get_pos()
                pos_x = (pos_x*size/resolution)+min_x
                pos_y = ((resolution-pos_y)*size/resolution)+min_y

                # Left click (zoom in)
                if event.button == 1:
                    size *= 0.5
                # Right click (zoom out)
                if event.button == 3:
                    size /= 0.5

                # Scale coords
                min_x = pos_x - size/2
                max_x = pos_x + size/2
                min_y = pos_y - size/2
                max_y = pos_y + size/2

                # Render and show zoomed image
                plot_fractal.render_image(
                    coords=(min_x, max_x, min_y, max_y),
                    iterations=iterations,
                    resolution=resolution,
                    fractal=fractal,
                    savename="./render/fractal.png")
                originalImg = pygame.image.load("./render/fractal.png")
                img = pygame.transform.scale(originalImg,
                                             (resolution, resolution))
                screen.blit(img, (0, 0))

        pygame.display.update()
        clock.tick(30)


if __name__ == '__main__':
    main()
