import numpy as np
import importlib
import os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import palettable


fractal_sets = {
    'mandelbrot',
    'burningship',
    'tricorn',
    'americo',
    # 'cosine',
    # 'sine',           } these three are not very good
    # 'exponential'
}


def render_image(
    coords=(-2.5, 1.5, -2.0, 2.0),
    iterations=1500,
    resolution=720,
    savename='./plot/fractal.png',
    fractal='mandelbrot'
                ):

    min_x, max_x, min_y, max_y = coords
    min_y, max_y = -max_y, -min_y
    coords = min_x, max_x, min_y, max_y

    # Image data set
    if fractal in fractal_sets:
        module_name = ('kernels.' + 'cuda_' + fractal)
        cuda_module = importlib.import_module(module_name)
    else:
        raise KeyError(f"Invalid fractal name. Select from: {fractal_sets}")

    # Colormapping

    """" ========= Example Colormaps ========= """
    # palettable.cubehelix.red_16
    # palettable.cubehelix.cubehelix1_16
    # palettable.cartocolors.sequential.agSunset_7
    # palettable.mycarta.LinearL_5
    # palettable.matplotlib.Magma_20
    # palettable.cubehelix.Cubehelix.make(start=0.3, rotation=-0.5, n=256)
    """" ===================================== """

    COLORMAP = palettable.cubehelix.cubehelix2_16
    COLORMAP_r = palettable.cubehelix.cubehelix2_16_r

    # Create colormap
    if iterations > 512:
        cmaps = []
        # Contrinually stack 2 colormaps of length 256 untl desired size
        for _ in range(iterations // 512):
            cmaps.append(
                COLORMAP.mpl_colormap(
                    np.linspace(0, 1, 256)))
            # Reversed colormaps used inbetween to make transition smooth
            cmaps.append(
                COLORMAP_r.mpl_colormap(
                    np.linspace(0, 1, 256)))
        cmaps = np.vstack(cmaps)
        cmap = LinearSegmentedColormap.from_list('cmap', cmaps)
        newcmap = cmap.from_list(
            'newcmap', list(map(cmap, range(256))), N=iterations-1)
        newcmap = cmap.from_list(
            # Add [0, 0, 0, 1] as to pixels inside the fractal set black
            'newcmap', list(map(cmap, range(256)))+[(0, 0, 0, 1)],
            N=iterations)
    # If iterations is below 512 use single colormap
    else:
        cmaps = COLORMAP.mpl_colormap(np.linspace(0, 1, iterations-1))
        cmap = LinearSegmentedColormap.from_list('cmap', cmaps)
        newcmap = cmap.from_list(
            # Add [0, 0, 0, 1] as to pixels inside the fractal set black
            'newcmap', list(map(cmap, range(256)))+[(0, 0, 0, 1)],
            N=iterations)

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

    # Default to .png if extension not given
    if '.' not in filename:
        filename += '.png'

    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save image
    Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(
        directory + "/" + filename)


if __name__ == '__main__':
    """" ========= Example Renders ========= """

    # render_image(fractal='burningship',
    #              savename='./screens/ship.png',
    #              coords=(-1.8,-1.7,-0.01,0.09),
    #              resolution=2000)

    # render_image(iterations=12,
    #              savename='./screens/simple_mandel.jpeg',
    #              resolution=3000)
