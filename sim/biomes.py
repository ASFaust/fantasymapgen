import numpy as np

# ── Biome table ──────────────────────────────────────────────────────────
#  ID  Name                      R    G    B
BIOME_TABLE = [
    ( 0, "Ocean",                 26,  82, 118),
    ( 1, "Ice Sheet",            232, 240, 248),
    ( 2, "Tundra",               140, 154, 126),
    ( 3, "Boreal Forest",         45,  90,  61),
    ( 4, "Cold Desert",          184, 169, 140),
    ( 5, "Temperate Grassland",  164, 184, 106),
    ( 6, "Temperate Forest",      74, 140,  92),
    ( 7, "Temperate Rainforest",  46, 110,  78),
    ( 8, "Hot Desert",           212, 196, 132),
    ( 9, "Savanna",              196, 180,  74),
    (10, "Tropical Seasonal",     58, 154,  74),
    (11, "Tropical Rainforest",   26, 106,  48),
    (12, "Alpine",               158, 158, 158),
]

BIOME_NAMES = [row[1] for row in BIOME_TABLE]
BIOME_COLORS = np.array(
    [[r, g, b] for (_, _, r, g, b) in BIOME_TABLE], dtype=np.uint8
)


def classify_biomes(
    temperature: np.ndarray,
    precipitation_mm: np.ndarray,
    elevation_km: np.ndarray,
    ocean_mask: np.ndarray,
    alpine_threshold_km: float = 2.5,
) -> np.ndarray:
    """Classify each pixel into a biome ID (uint8)."""

    out = np.zeros(temperature.shape, dtype=np.uint8)

    T = temperature
    P = precipitation_mm

    # ─────────────────────────────────────────────
    # Ocean
    # ─────────────────────────────────────────────
    out[ocean_mask] = 0
    land = ~ocean_mask

    # ─────────────────────────────────────────────
    # Cold bands
    # ─────────────────────────────────────────────
    out[land & (T < -15)] = 1                           # Ice Sheet
    out[land & (T >= -15) & (T < -5)] = 2               # Tundra

    band = land & (T >= -5) & (T < 7)                   # Cool boreal band
    out[band & (P >= 300)] = 3                          # Boreal Forest
    out[band & (P < 300)]  = 4                          # Cold Desert

    # ─────────────────────────────────────────────
    # Temperate 7–10°C (transition band)
    # ─────────────────────────────────────────────
    band = land & (T >= 7) & (T < 10)
    out[band & (P < 250)]                 = 4           # Cold Desert
    out[band & (P >= 250) & (P < 800)]    = 5           # Temperate Grassland
    out[band & (P >= 800) & (P <= 1800)]  = 6           # Temperate Forest
    out[band & (P > 1800)]                = 7           # Temperate Rainforest

    # ─────────────────────────────────────────────
    # Temperate 10–20°C
    # ─────────────────────────────────────────────
    band = land & (T >= 10) & (T < 20)
    out[band & (P < 250)]                 = 8           # Hot Desert
    out[band & (P >= 250) & (P < 800)]    = 5           # Temperate Grassland
    out[band & (P >= 800) & (P <= 1800)]  = 6           # Temperate Forest
    out[band & (P > 1800)]                = 7           # Temperate Rainforest

    # ─────────────────────────────────────────────
    # Tropical ≥20°C
    # ─────────────────────────────────────────────
    band = land & (T >= 20)
    out[band & (P < 250)]                 = 8           # Hot Desert
    out[band & (P >= 250) & (P < 1000)]   = 9           # Savanna
    out[band & (P >= 1000) & (P <= 2200)] = 10          # Tropical Seasonal
    out[band & (P > 2200)]                = 11          # Tropical Rainforest

    # ─────────────────────────────────────────────
    # Alpine override
    # ─────────────────────────────────────────────
    out[land & (elevation_km > alpine_threshold_km)] = 12

    return out