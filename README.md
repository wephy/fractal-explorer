Interactive Fractal Explorer
==============================

![Version 0.2.1](https://img.shields.io/badge/version-0.2.1-blue)

Features
--------
* Interactive
* Cuda processing
* Colormapping
* Multiple fractals:
  * [Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set)
  * [Burning ship](https://en.wikipedia.org/wiki/Burning_Ship_fractal)

Executing
--------
To run the program, execute `main()` within main.py

All parameters are optional. The defaults are:
`fractal='mandelbrot'`
`iterations=1500`
`image_size=1000`
`coords=(-2.5, 1.5, 2.0, 2.0)`

Here is an example of executing with custom parameters:
```python
    main(fractal='burningship',
         iterations=2500,
         image_size=720,
         coords=(-1.8, -1.7, -0.01, 0.09))
```
> NOTE: All testing so far has been through IPython within VSCode

Controls
--------
##### (All controls take place on cursor position)
* Left-click: zoom in
* Right-click: zoom out
* Middle-mouse: relocate center

Requirements
------------
Python 3.8 or later

For module requirements see [requirements.txt](https://github.com/wephy/py-fractals/blob/main/requirements.txt)

To-do
------
* Implement command-line execution
* Add more fractal sets
* Add smooth zoom animation
* Add regular jit and/or cuda emulation fallback for cuda incompatible systems
* Add menu screen (for entering fractal, resolution, colormap, max iterations, etc.) => turn into executable
* Add additional controls (e.g. return to original position)

License
-------
py-fractal is licensed under the terms of the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0)
