import pygame
import sys
from pygame.locals import *
from plot_fractal import render_image


def main(fractal='mandelbrot', iterations=1500, resolution=1000,
         coords=(-2.5, 1.5, -2.0, 2.0)):

    # Create coord variables
    min_x, max_x, min_y, max_y = coords
    size = max_x - min_x

    # Render first image (Executed first as a check)
    render_image(coords=coords, iterations=iterations,
                 resolution=resolution, fractal=fractal)

    # Pygame setup
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption('Fractal Plot')
    screen = pygame.display.set_mode((resolution, resolution))
    pygame.display.set_caption(f'Fractal plot: {fractal}')

    # Load and show first image
    img = pygame.image.load("./fractal.png")
    screen.blit(img, (0, 0))

    # Render zoomed image on click
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
                render_image(coords=(min_x, max_x, min_y, max_y),
                             iterations=iterations,
                             resolution=resolution,
                             fractal=fractal
                             )
                originalImg = pygame.image.load("./fractal.png")
                img = pygame.transform.scale(originalImg,
                                             (resolution, resolution))
                screen.blit(img, (0, 0))

        pygame.display.update()
        clock.tick(30)


if __name__ == '__main__':
    """
    Fractals:
    - mandelbrot
    - burningship
    """

    # main()

    # main(fractal='burningship')

    # main(resolution=360)

    # main(fractal='burningship', coords=(-1.8, -1.7, -0.01, 0.09), iterations=3000)
