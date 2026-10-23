# Reference: runevision erosion filter

Reference material for the terrain generation, not used by the code yet.

`erosion_filter.glsl` is the heightmap buffer of Rune Skovbo Johansen's
"Advanced Terrain Erosion Filter" Shadertoy, saved verbatim. It contains:

- **Phacelle noise**: stripes aligned with a given direction, blended over a 4x4 grid of jittered cells.
- **ErosionFilter**: stacks octaves of Phacelle gullies that follow the downhill slope, fading them
  out on flat ridges and creases. It returns a height and slope delta plus a ridge map
  (-1 on creases, 1 on ridges).
- **Demonstration**: the `Heightmap()` function, with the parameter values and explanations.

Sources:

- Blog post: https://blog.runevision.com/2026/03/fast-and-gorgeous-erosion-filter.html
- Video: https://www.youtube.com/watch?v=gsJHzBTPG0Y

The snippet uses helpers from the Shadertoy's Common tab that are not included here: `hash`
(vec2 to vec2 in [-1, 1]), `noised` (value or gradient noise returning height plus derivatives),
`clamp01`, `pack4`, `AnimateLoHi` and `AnimateWaveTo`, and constants such as `GRASS_HEIGHT`,
`WATER_HEIGHT`, `BUFFER_SIZE` and `DISCARD_MAP`.

The Phacelle noise and erosion functions are copyright (c) 2025 Rune Skovbo Johansen and licensed
under the Mozilla Public License 2.0 (https://mozilla.org/MPL/2.0/). Under the MPL, those functions
stay under the MPL in any port, with their copyright notice kept.
