"""
Slope Analysis Module
=====================
Calculate slope and aspect from Digital Elevation Models.
"""

import numpy as np
from rasterio.transform import Affine
from typing import Tuple, Optional


def calculate_slope(dem: np.ndarray, transform: Affine, 
                    units: str = 'degrees') -> np.ndarray:
    """
    Calculate slope from a DEM using the Horn algorithm.
    
    The Horn algorithm uses a 3x3 window to calculate the slope
    at each cell based on the elevation of its 8 neighbors.
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        units: Output units - 'degrees', 'radians', or 'percent'
        
    Returns:
        2D numpy array of slope values
    """
    # Get cell size from transform
    cell_size_x = abs(transform[0])
    cell_size_y = abs(transform[4])
    cell_size = (cell_size_x + cell_size_y) / 2  # Average for non-square cells
    
    # Pad the array to handle edges
    padded = np.pad(dem, 1, mode='edge')
    
    # Extract the 8 neighbors using array slicing
    # a b c
    # d e f  where e is the center cell
    # g h i
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]
    
    # Calculate dz/dx and dz/dy using Horn's method
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_size)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_size)
    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * cell_size)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * cell_size)
    
    # Calculate slope
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    
    # Convert to requested units
    if units == 'degrees':
        slope = np.degrees(slope_rad)
    elif units == 'percent':
        slope = np.tan(slope_rad) * 100
    else:  # radians
        slope = slope_rad
    
    # Handle NaN values
    slope[np.isnan(dem)] = np.nan
    
    return slope.astype(np.float32)


def calculate_aspect(dem: np.ndarray, transform: Affine,
                     flat_value: float = -1.0) -> np.ndarray:
    """
    Calculate aspect (direction of steepest descent) from a DEM.
    
    Aspect is measured clockwise from north:
    - North = 0° (or 360°)
    - East = 90°
    - South = 180°
    - West = 270°
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        flat_value: Value to assign to flat areas (default -1)
        
    Returns:
        2D numpy array of aspect values in degrees
    """
    # Get cell size from transform
    cell_size_x = abs(transform[0])
    cell_size_y = abs(transform[4])
    cell_size = (cell_size_x + cell_size_y) / 2
    
    # Pad the array to handle edges
    padded = np.pad(dem, 1, mode='edge')
    
    # Extract neighbors
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]
    
    # Calculate gradients
    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * cell_size)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * cell_size)
    
    # Calculate aspect
    # atan2 gives angle from -π to π, we convert to 0-360 degrees
    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect = np.degrees(aspect_rad)
    
    # Convert to compass direction (clockwise from north)
    aspect = 90 - aspect
    aspect[aspect < 0] += 360
    aspect[aspect >= 360] -= 360
    
    # Handle flat areas
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    flat_mask = np.degrees(slope_rad) < 0.01
    aspect[flat_mask] = flat_value
    
    # Handle NaN values
    aspect[np.isnan(dem)] = np.nan
    
    return aspect.astype(np.float32)


def calculate_curvature(dem: np.ndarray, transform: Affine,
                        curvature_type: str = 'total') -> np.ndarray:
    """
    Calculate surface curvature from a DEM.
    
    Curvature describes the shape of the surface:
    - Positive values = convex (peaks, ridges)
    - Negative values = concave (valleys, depressions)
    - Near zero = flat or linear slope
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        curvature_type: 'total', 'plan' (horizontal), or 'profile' (vertical)
        
    Returns:
        2D numpy array of curvature values
    """
    cell_size = abs(transform[0])
    
    # Pad array
    padded = np.pad(dem, 1, mode='edge')
    
    # Extract neighbors
    z1 = padded[:-2, :-2]   # upper left
    z2 = padded[:-2, 1:-1]  # upper center
    z3 = padded[:-2, 2:]    # upper right
    z4 = padded[1:-1, :-2]  # middle left
    z5 = padded[1:-1, 1:-1] # center
    z6 = padded[1:-1, 2:]   # middle right
    z7 = padded[2:, :-2]    # lower left
    z8 = padded[2:, 1:-1]   # lower center
    z9 = padded[2:, 2:]     # lower right
    
    # Calculate second derivatives
    d2z_dx2 = (z4 - 2*z5 + z6) / (cell_size**2)
    d2z_dy2 = (z2 - 2*z5 + z8) / (cell_size**2)
    d2z_dxdy = (z3 - z1 - z9 + z7) / (4 * cell_size**2)
    
    # First derivatives for profile/plan curvature
    dz_dx = (z6 - z4) / (2 * cell_size)
    dz_dy = (z8 - z2) / (2 * cell_size)
    
    if curvature_type == 'total':
        # Laplacian (total curvature)
        curvature = d2z_dx2 + d2z_dy2
    
    elif curvature_type == 'profile':
        # Profile curvature (in direction of steepest slope)
        p = dz_dx**2 + dz_dy**2
        p[p == 0] = 1e-10  # Avoid division by zero
        curvature = -((d2z_dx2 * dz_dx**2 + 2*d2z_dxdy*dz_dx*dz_dy + d2z_dy2*dz_dy**2) / 
                      (p * np.sqrt(1 + p)))
    
    elif curvature_type == 'plan':
        # Plan curvature (perpendicular to slope direction)
        p = dz_dx**2 + dz_dy**2
        p[p == 0] = 1e-10
        curvature = -((d2z_dx2 * dz_dy**2 - 2*d2z_dxdy*dz_dx*dz_dy + d2z_dy2*dz_dx**2) / 
                      (p**1.5))
    
    else:
        raise ValueError(f"Unknown curvature type: {curvature_type}")
    
    # Scale to standard units (typically *100)
    curvature = curvature * 100
    
    # Handle NaN values
    curvature[np.isnan(dem)] = np.nan
    
    return curvature.astype(np.float32)


def calculate_roughness(dem: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Calculate terrain roughness index.
    
    Roughness is the standard deviation of elevation within a moving window.
    Higher values indicate more rugged terrain.
    
    Args:
        dem: 2D numpy array of elevation values
        window_size: Size of the moving window (must be odd)
        
    Returns:
        2D numpy array of roughness values
    """
    from scipy import ndimage
    
    if window_size % 2 == 0:
        window_size += 1
    
    # Calculate local mean
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    local_mean = ndimage.convolve(dem, kernel, mode='reflect')
    
    # Calculate local variance
    local_sq_mean = ndimage.convolve(dem**2, kernel, mode='reflect')
    local_variance = local_sq_mean - local_mean**2
    local_variance[local_variance < 0] = 0  # Handle numerical errors
    
    # Standard deviation = roughness
    roughness = np.sqrt(local_variance)
    
    # Handle NaN values
    roughness[np.isnan(dem)] = np.nan
    
    return roughness.astype(np.float32)


def calculate_tri(dem: np.ndarray) -> np.ndarray:
    """
    Calculate Terrain Ruggedness Index (TRI).
    
    TRI is the mean difference between a central pixel and its surrounding cells.
    
    Args:
        dem: 2D numpy array of elevation values
        
    Returns:
        2D numpy array of TRI values
    """
    # Pad array
    padded = np.pad(dem, 1, mode='edge')
    
    # Calculate sum of squared differences from center cell
    center = dem
    
    # All 8 neighbors
    neighbors = [
        padded[:-2, :-2],   # NW
        padded[:-2, 1:-1],  # N
        padded[:-2, 2:],    # NE
        padded[1:-1, :-2],  # W
        padded[1:-1, 2:],   # E
        padded[2:, :-2],    # SW
        padded[2:, 1:-1],   # S
        padded[2:, 2:]      # SE
    ]
    
    # Calculate squared differences
    sum_sq_diff = np.zeros_like(dem)
    for neighbor in neighbors:
        sum_sq_diff += (neighbor - center)**2
    
    # TRI = sqrt(mean of squared differences)
    tri = np.sqrt(sum_sq_diff / 8)
    
    # Handle NaN values
    tri[np.isnan(dem)] = np.nan
    
    return tri.astype(np.float32)


def classify_slope(slope: np.ndarray, 
                   classes: Optional[list] = None) -> np.ndarray:
    """
    Classify slope into categories.
    
    Args:
        slope: 2D numpy array of slope values in degrees
        classes: List of break values for classification
                 Default: [0, 3, 8, 15, 25, 35, 45, 90]
        
    Returns:
        2D numpy array of slope class values (1 to n classes)
    """
    if classes is None:
        # Standard slope classes (degrees)
        classes = [0, 3, 8, 15, 25, 35, 45, 90]
    
    classified = np.zeros_like(slope, dtype=np.uint8)
    
    for i in range(len(classes) - 1):
        mask = (slope >= classes[i]) & (slope < classes[i + 1])
        classified[mask] = i + 1
    
    # Handle NaN values
    classified[np.isnan(slope)] = 0
    
    return classified
