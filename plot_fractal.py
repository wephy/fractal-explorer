import numpy as np
import importlib
import os
from PIL import Image
from matplotlib import colors as mc
import palettable


def render_image(coords=(-2.5, 1.5, -2.0, 2.0), iterations=1500,
                 resolution=1000, savename='fractal', fractal='mandelbrot'):

    min_x, max_x, min_y, max_y = coords
    min_y, max_y = -max_y, -min_y
    coords = min_x, max_x, min_y, max_y

    fractals = {'mandelbrot', 'burningship'}

    if iterations < 512:
        raise ValueError("Minimum 512 iterations")

    # Image data set
    if fractal in fractals:
        module_name = ('kernels.' + 'cuda_' + fractal)
        cuda_module = importlib.import_module(module_name)
    else:
        raise KeyError(f"Invalid fractal name. Select from: {fractals}")

    # Colormapping
    cmaps = []
    for _ in range(iterations // 512):
        cmaps.append(
            palettable.matplotlib.Magma_20.mpl_colormap(
                np.linspace(0, 1, 256)))
        cmaps.append(
            palettable.matplotlib.Magma_20_r.mpl_colormap(
                np.linspace(0, 1, 256)))
    cmaps = np.vstack(cmaps)
    cmap = mc.LinearSegmentedColormap.from_list('cmap', cmaps)
    newcmap = cmap.from_list(
        'newcmap', list(map(cmap, range(255))), N=iterations-1)
    newcmap = cmap.from_list(
        'newcmap', list(map(cmap, range(255)))+[(0, 0, 0, 1)], N=iterations)

    """" ========= Example Colormaps ========= """
    # palettable.cubehelix.red_16.mpl_colormap
    # palettable.cubehelix.cubehelix1_16.mpl_colormap
    # palettable.cartocolors.sequential.agSunset_7.mpl_colormap
    # palettable.mycarta.LinearL_5.mpl_colormap
    """" ===================================== """

    # Get image from cuda_mandelbrot
    image = cuda_module.get_image(
        coords=coords,
        max_iters=iterations,
        resolution=resolution)

    # Apply colormap
    colored_image = newcmap(image)

    # Turn into image
    directory = '/'.join(savename.split("/")[:-1])
    if directory == '':
        directory += './'
    filename = savename.split("/")[-1]
    if '.' not in filename:
        filename += '.png'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save image
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(
        directory + "/" + filename)


if __name__ == '__main__':
    """" ========= Example Renders ========= """

    # render_image(fractal='mandelbrot',
    #              savename='./screens/mandelbrot.png',
    #              coords=(-0.74, -0.67, 0.21, 0.28),
    #              resolution=4*1920)

    
    # render_image(fractal='burningship',
    #              savename='./screens/ship.png',
    #              coords=(-2.444, 2.0, -0.5, 2.0),
    #              resolution=20000)
