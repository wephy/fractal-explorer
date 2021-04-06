import pygame
import sys
from pygame.locals import *
from plot_fractal import render_image


def main(fractal='mandelbrot', iterations=1500, image_size=1000,
         coords=(-2.5, 1.5, -2.0, 2.0), size=4):

    # Create coord variables
    min_x, max_x, min_y, max_y = coords
    size = max_x - min_x

    # Render first image (Executed first as a check)
    render_image(coords=coords, iterations=iterations,
                 resolution=image_size, fractal=fractal)

    # Pygame setup
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption('Fractal Plot')
    screen = pygame.display.set_mode((image_size, image_size))
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
                pos_x = (pos_x*size/image_size)+min_x
                pos_y = ((image_size-pos_y)*size/image_size)+min_y

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
                             resolution=image_size,
                             fractal=fractal
                             )
                originalImg = pygame.image.load("./fractal.png")
                img = pygame.transform.scale(originalImg,
                                             (image_size, image_size))
                screen.blit(img, (0, 0))

        pygame.display.update()
        clock.tick(30)


if __name__ == '__main__':
    """
    Fractals:
    - mandelbrot
    - burningship
    """

    main(
        fractal='burningship',
        iterations=1500,
        image_size=1000,
        coords=(-1.8, -1.7, -0.01, 0.09)
         )
