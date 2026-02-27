# mapstuff

A fantasy map generator for whole earth-like planets. Combines IQ-style domain-warped Perlin noise with a static weather simulation to produce realistic-looking worlds, viewable in both 2D and 3D.

![mapstuff screenshot](screenshot.png)

## Features

- **Procedural terrain** using Inigo Quilez's double-warp domain warping over fractional Brownian motion (fBm) noise, sampled on a sphere to avoid pole and seam artifacts
- **Land/ocean ratio control** with configurable polar exclusion zones
- **Interactive curve editor** for non-linear height remapping
- **Static weather simulation**
  - Global wind circulation (trade winds, westerlies, polar easterlies)
  - Temperature from latitude insolation + elevation lapse rate (6.5 °C/km)
  - Precipitation: ocean moisture decay, onshore wind alignment, orographic lift (semi-Lagrangian advection)
  - Ice cap overlay driven by temperature
- **2D map view** with wind vector overlay
- **3D sphere view** — texture-mapped OpenGL globe with mouse rotate/zoom
- **Layer selection** — heightmap, temperature, or precipitation with matplotlib colormaps
- **Preset system** — save and load configurations as YAML files

## Stack

| Purpose | Library |
|---|---|
| GUI | PyQt6 |
| 3D rendering | PyOpenGL |
| Numerics | NumPy, SciPy |
| Noise (JIT) | Numba |
| Colormaps | Matplotlib |
| Image I/O | Pillow |
| Config | PyYAML |

## Installation

```bash
pip install PyQt6 PyOpenGL PyOpenGL_accelerate numpy scipy matplotlib pillow numba pyyaml
```

Python 3.10+ recommended.

## Running

```bash
python main.py
```

## Usage

1. Adjust terrain sliders (size, seed, octaves, frequency, warp strengths, land %, polar exclusion)
2. Optionally edit the height remap curve (left-click to add points, right-click to remove)
3. Hit **Generate** to produce a new heightmap
4. Toggle between **2D** and **3D** view
5. Select a base layer: **Heightmap**, **Temperature**, or **Precipitation**
6. Toggle **Wind** and **Ice** overlays
7. Click **Calculate Weather** to run the weather simulation
8. Save/load presets from the terrain panel

## Project Structure

```
gen/        procedural noise and heightmap generation
render/     layer compositor and colormap rendering
sim/        static weather simulation (wind, temperature, precipitation)
ui/         PyQt6 GUI (main window, OpenGL widget, curve editor, etc.)
main.py     entry point
```

## References

- [Inigo Quilez — Domain Warping](https://iquilezles.org/articles/warp/) — the core terrain warping technique
