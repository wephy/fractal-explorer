import numpy as np
import importlib
import os
from PIL import Image
# from matplotlib import pyplot as plt
from matplotlib import colors as mc
# import colorcet as cc
import palettable
# from palettable.cubehelix import Cubehelix


def render_image(
    coords=(-2.5, 1.5, -2.0, 2.0), max_iters=1500, resolution=1000,
    save_name='fractal', fractal='mandelbrot'):

    fractals = {'mandelbrot', 'burningship'}

    # Image data set
    if fractal in fractals:
        module_name = ('kernels.' + 'cuda_' + fractal)
        cuda_module = importlib.import_module(module_name)
    else:
        raise KeyError(f"Invalid fractal name. Select from: {fractals}")

    # Iterations
    ITERATIONS = max_iters

    """" ========= Example Colormaps ========= """
    # palettable.cubehelix.red_16.mpl_colormap
    # palettable.cubehelix.cubehelix1_16.mpl_colormap
    # palettable.cartocolors.sequential.agSunset_7.mpl_colormap
    # palettable.mycarta.LinearL_5.mpl_colormap

    # Colormapping
    cmaps = []
    for _ in range(ITERATIONS // 512):
        cmaps.append(palettable.matplotlib.Magma_20.mpl_colormap(np.linspace(0., 1, 256)))
        cmaps.append(palettable.matplotlib.Magma_20_r.mpl_colormap(np.linspace(0., 1, 256)))
    cmaps = np.vstack(cmaps)
    cmap = mc.LinearSegmentedColormap.from_list('cmap', cmaps)
    newcmap = cmap.from_list('newcmap',list(map(cmap,range(255))), N=ITERATIONS-1)
    newcmap = cmap.from_list('newcmap',list(map(cmap,range(255)))+[(0,0,0,1)], N=ITERATIONS)

    # Get image from cuda_mandelbrot
    image = cuda_module.get_image(
        coords=coords,
        max_iters=max_iters,
        resolution=resolution
    )

    # Apply colormap
    colored_image = newcmap(image)

    # Turn into image
    directory = '/'.join(save_name.split("/")[:-1])
    if directory == '':
        directory += './'
    filename = save_name.split("/")[-1]
    if '.' not in filename:
        filename += '.png'
    if not os.path.exists(directory):
        os.makedirs(directory)

    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(directory + "/" + filename)


if __name__ == '__main__':
    render_image(fractal='mandelbrot', save_name='./screens/mandelbrot.jpeg')
    render_image(fractal='burningship', save_name='./screens/burningship')