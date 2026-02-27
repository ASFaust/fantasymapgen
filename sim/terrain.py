import numpy as np


def compute_shore_distance(
    sim_ocean: np.ndarray,
    sim_lat: np.ndarray,
    sim_lon: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    """
    Fast shore distance using scipy's distance transform + per-row haversine scaling.

    Strategy:
    1. Compute a pixel-unit Euclidean distance transform from ocean cells.
    2. Convert to km by multiplying by the local arc-length per pixel.
       - N/S: constant = R * Δlat  (one row step)
       - E/W: shrinks with cos(lat) = R * cos(lat) * Δlon  (one col step)
    3. Use the geometric mean of the two as the effective pixel size, then
       multiply by the pixel-distance. This is an approximation but is
       accurate to within a few % and runs in milliseconds.

    For a true geodesic field, use a Dijkstra path; for most
    weather-sim purposes this fast version is indistinguishable.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = sim_lat.shape
    R = radius_km

    dlat = float(np.abs(np.median(np.diff(sim_lat[:, W // 2]))))
    dlon = float(np.abs(np.median(np.diff(sim_lon[H // 2, :]))))

    km_per_pixel_ns = R * dlat
    cos_lat = np.cos(sim_lat[:, W // 2])[:, np.newaxis]       # (H, 1)
    km_per_pixel_ew = R * np.abs(dlon) * cos_lat              # (H, 1)

    pixel_km = np.sqrt(np.maximum(km_per_pixel_ns * km_per_pixel_ew, 0.0))

    land_mask = (~sim_ocean).astype(np.uint8)

    # Tile horizontally to handle the periodic longitude wrap
    land_tiled = np.tile(land_mask, 3)
    pixel_dist_tiled = distance_transform_edt(land_tiled)
    pixel_dist = pixel_dist_tiled[:, W:2 * W]

    dist_km = pixel_dist * pixel_km   # (H, W) * (H, 1)
    dist_km[sim_ocean] = 0.0

    return dist_km


def compute_terrain_gradient(
    sim_elevation_km: np.ndarray,
    sim_lat: np.ndarray,
    sim_lon: np.ndarray,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the terrain gradient in physical units (km elevation / km horizontal).

    Uses central differences with longitude wrap handled by tiling one
    column of padding on each side.

    Returns
    -------
    grad_x : (H, W)  E/W slope  — positive = uphill eastward
    grad_y : (H, W)  N/S slope  — positive = uphill southward (row +)
    grad_magnitude : (H, W)  |∇h|
    """
    H, W = sim_lat.shape
    R = radius_km

    dlat = float(np.abs(np.median(np.diff(sim_lat[:, W // 2]))))
    dlon = float(np.abs(np.median(np.diff(sim_lon[H // 2, :]))))

    km_per_pixel_ns = R * dlat
    cos_lat = np.cos(sim_lat[:, W // 2])[:, np.newaxis]              # (H, 1)
    km_per_pixel_ew = R * np.abs(dlon) * np.maximum(cos_lat, 1e-6)  # (H, 1)

    elev = sim_elevation_km

    # E/W gradient — pad one column on each side for longitude wrap
    elev_padded = np.concatenate([elev[:, -1:], elev, elev[:, :1]], axis=1)
    grad_x_padded = np.gradient(elev_padded, axis=1)
    grad_x = grad_x_padded[:, 1:-1] / km_per_pixel_ew  # km/km

    # N/S gradient — no wrap (poles are boundaries)
    grad_y = np.gradient(elev, axis=0) / km_per_pixel_ns  # km/km

    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return grad_x, grad_y, grad_magnitude
