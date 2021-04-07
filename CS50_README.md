Interactive Fractal Explorer
============================
Name: Joseph Webb
Country: United Kingdom
GitHub: wephy
Video Demo:  https://www.youtube.com/watch?v=c2XnuJAZGMg
Website: wephy.com

### Project Goal
------------
Create an interactive fractal generator in which you zoom in and out to explore.

### Structure
---------
The project, at the moment, contains 4 `.py` files.  


#### `cuda_<fractal>.py` (2 files)
Two of the four are 'cuda kernels'. These are files which 'generate' a particular fractal. They use cuda processing---this was probably where I struggled the most in this project, requiring the largest amount research (thankfully I found some amazing open source work [here](https://www.kaggle.com/landlord/numba-cuda-mandelbrot/execution) which helped massively). The output of the kernels is an array whereby each value is the iteration count of a particular a pixel. These files take inputs `coords, max_iters, resolution` on the `render_image()` function  which then executes the two other function in the file (decorated with @cuda.jit).
This part of the project was incredibly educational in regards to scientific computing; it exposed me to using numba properly, declaring data types in python and working with cuda processing.
> Notes: I plan on adding more kernels for different fractal patterns and implementing regular @jit fallback and/or cuda emulation for cuda incompatible systems.

#### `plot_fractal.py`
This file can be used standalone, but is also fundamental for `main.py`. It's purpose is to execute a respective `cuda_<fractal>` kernel (depending on the `fractal` parameter) and produce an image from the output. It uses matplotlib and palettable to apply colormaps, importlib to import only the required kernel, PIL and os libraries to save the final image to a desired location. It is comprised of one function `render_image()`, which takes the same inputs as the kernel files, with the added parameter for `savename` which can be used to save to any folder (it will create the directory if it doesn't exist), and to any image file extension [supported by PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) (the default is `.png`).
> Notes: Colormaping is currently applied by changing two lines of code manually - I plan on making this a parameter whereby any colormap from matplotlib/palettable can be used.

#### `main.py`
This file is the main executable for running the interactive program. Fundamentally built with pygame, it uses `render_image` from `plot_fractal.py` to save images and load them. It takes mouse controls to zoom in and out of the fractal---doing math on the mouse position and co-ordinates of the image to then supply these create new `coords` to send to `render_image()`. Like in the previous files, the `main()` function in this file can take the same parameters to get a different window size, max iterations or starting position.
> Notes: This file has the most potential, where possibilites include: implementing command-line execution, smooth zoom animations, additional controls and eventually a menu screen packaged in a `.exe`.

### Project challenges and decisions
--------------------------------
This project was an amazing experience to really explore and uitilise GitHub; although, this provided its fair share of hurdles: understanding proper app/file structure (which I believe I still have lots of progress to make), creating `.gitignore`, `README.md`, `LICENSE` and `requirements.txt` files, and most importantly getting used to git commands such as `add`, `commit` and `push`.
Other challenges include using unfamiliar libraries, such as: pygame and numba; also applying concepts such as: colormaping, normalising (which I ultimately discarded in favour of colormap stacking), dynamically importing, optimising and statically typing in Python.
I opted for cuda processing because it was a new area for me and I believe it will be applicable for me in the future (scientific computing).
Another decision, which I'm not quite sure if I made the correct one, was to go with pygame. I looked into alternatives, but pygame, for my use case, seemed the most simplistic. 


### [See full project on GitHub](https://github.com/wephy/fractal-explorer)