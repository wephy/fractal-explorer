import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda
from numba import uint32, f8, uint16


@cuda.jit(uint32(f8, f8, uint32), device=True)
def mandel(x, y, max_iters):
    c = complex(x, y)
    z = 0j
    lz = 0j
    for i in range(max_iters):
        z = z*z + lz + c
        lz = z
        if (z.real*z.real + z.imag*z.imag) >= 16:
            return i
    return max_iters


@cuda.jit((f8, f8, f8, f8, uint16[:, :], uint32))
def mandel_kernel(min_x, max_x, min_y, max_y, image, max_iters):
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, max_iters)


def get_image(coords, max_iters, resolution):
    min_x, max_x, min_y, max_y = coords
    x_diff = max_x - min_x
    y_diff = max_y - min_y
    if x_diff > y_diff:
        width = resolution
        height = y_diff / x_diff * resolution
    else:
        height = resolution
        width = x_diff / y_diff * resolution

    gimage = np.zeros((int(height), int(width)), dtype=np.uint16)
    blockdim = (32, 8)
    griddim = (32, 16)
    d_image = cuda.to_device(gimage)
    mandel_kernel[griddim, blockdim](
        min_x, max_x, min_y, max_y, d_image, max_iters)
    h_image = d_image.copy_to_host()
    return h_image


if __name__ == '__main__':
    # Check execution time of image kernel
    start = timer()
    image = get_image()
    dt = timer() - start
    print("Mandelbrot created on GPU in %f s" % dt)
    imshow(image)
    show()
