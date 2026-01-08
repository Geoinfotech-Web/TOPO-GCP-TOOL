"""
DEM Processor Module
====================
Handles loading, processing, and saving of Digital Elevation Model data.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from typing import Tuple, Optional, Union
import os


class DEMProcessor:
    """
    A class to handle DEM file operations including loading, processing, and saving.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the DEM processor with a file path.
        
        Args:
            filepath: Path to the DEM file (GeoTIFF format)
        """
        self.filepath = filepath
        self.dem_data: Optional[np.ndarray] = None
        self.transform: Optional[Affine] = None
        self.crs: Optional[CRS] = None
        self.bounds: Optional[Tuple] = None
        self.nodata: Optional[float] = None
        self.profile: Optional[dict] = None
    
    def load_dem(self) -> Tuple[np.ndarray, Affine, CRS, Tuple]:
        """
        Load DEM data from the file.
        
        Returns:
            Tuple containing:
                - dem_data: 2D numpy array of elevation values
                - transform: Affine transformation matrix
                - crs: Coordinate reference system
                - bounds: Bounding box (left, bottom, right, top)
        """
        with rasterio.open(self.filepath) as src:
            self.dem_data = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.nodata = src.nodata
            self.profile = src.profile.copy()
            
            # Handle nodata values
            if self.nodata is not None:
                self.dem_data[self.dem_data == self.nodata] = np.nan
            
            # Also handle common nodata values
            self.dem_data[self.dem_data < -9000] = np.nan
            self.dem_data[self.dem_data > 9000] = np.nan
        
        return self.dem_data, self.transform, self.crs, self.bounds
    
    def get_resolution(self) -> Tuple[float, float]:
        """
        Get the spatial resolution of the DEM.
        
        Returns:
            Tuple of (x_resolution, y_resolution) in map units
        """
        if self.transform is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        return abs(self.transform[0]), abs(self.transform[4])
    
    def get_extent(self) -> Tuple[float, float, float, float]:
        """
        Get the geographic extent of the DEM.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if self.bounds is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        return (self.bounds.left, self.bounds.bottom, 
                self.bounds.right, self.bounds.top)
    
    def get_statistics(self) -> dict:
        """
        Calculate basic statistics for the DEM.
        
        Returns:
            Dictionary containing min, max, mean, std elevation values
        """
        if self.dem_data is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        valid_data = self.dem_data[~np.isnan(self.dem_data)]
        
        return {
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'range': float(np.max(valid_data) - np.min(valid_data))
        }
    
    def save_raster(self, data: np.ndarray, output_path: str, 
                    layer_name: str = "output",
                    dtype: str = 'float32') -> str:
        """
        Save a raster array to a GeoTIFF file.
        
        Args:
            data: 2D numpy array to save
            output_path: Path for the output file
            layer_name: Name for the layer
            dtype: Data type for the output
            
        Returns:
            Path to the saved file
        """
        if self.profile is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        # Update profile for output
        profile = self.profile.copy()
        profile.update(
            dtype=dtype,
            count=1,
            compress='lzw'
        )
        
        # Handle different data types
        if dtype == 'int32':
            data = np.nan_to_num(data, nan=-9999).astype(np.int32)
            profile['nodata'] = -9999
        elif dtype == 'uint8':
            data = np.nan_to_num(data, nan=0).astype(np.uint8)
            profile['nodata'] = 0
        else:
            data = data.astype(np.float32)
            profile['nodata'] = np.nan
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(1, layer_name=layer_name)
        
        return output_path
    
    def pixel_to_coords(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            row: Row index (y pixel)
            col: Column index (x pixel)
            
        Returns:
            Tuple of (x, y) in map coordinates
        """
        if self.transform is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        x = self.transform.c + col * self.transform.a + row * self.transform.b
        y = self.transform.f + col * self.transform.d + row * self.transform.e
        
        return x, y
    
    def coords_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            x: X coordinate in map units
            y: Y coordinate in map units
            
        Returns:
            Tuple of (row, col) pixel indices
        """
        if self.transform is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        # Inverse transformation
        inv_transform = ~self.transform
        col, row = inv_transform * (x, y)
        
        return int(row), int(col)
    
    def get_elevation_at_point(self, x: float, y: float, 
                                interpolate: bool = True) -> float:
        """
        Get elevation at a specific geographic point.
        
        Args:
            x: X coordinate
            y: Y coordinate
            interpolate: Whether to use bilinear interpolation
            
        Returns:
            Elevation value at the point
        """
        if self.dem_data is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        row, col = self.coords_to_pixel(x, y)
        
        # Check bounds
        if (row < 0 or row >= self.dem_data.shape[0] or 
            col < 0 or col >= self.dem_data.shape[1]):
            return np.nan
        
        if not interpolate:
            return float(self.dem_data[row, col])
        
        # Bilinear interpolation
        inv_transform = ~self.transform
        col_f, row_f = inv_transform * (x, y)
        
        row0, col0 = int(row_f), int(col_f)
        row1, col1 = min(row0 + 1, self.dem_data.shape[0] - 1), min(col0 + 1, self.dem_data.shape[1] - 1)
        
        # Weights
        wr = row_f - row0
        wc = col_f - col0
        
        # Interpolate
        z = (self.dem_data[row0, col0] * (1 - wr) * (1 - wc) +
             self.dem_data[row0, col1] * (1 - wr) * wc +
             self.dem_data[row1, col0] * wr * (1 - wc) +
             self.dem_data[row1, col1] * wr * wc)
        
        return float(z)
    
    def resample(self, target_resolution: float, 
                 method: str = 'bilinear') -> np.ndarray:
        """
        Resample the DEM to a different resolution.
        
        Args:
            target_resolution: Desired resolution in map units
            method: Resampling method ('nearest', 'bilinear', 'cubic')
            
        Returns:
            Resampled DEM array
        """
        from scipy import ndimage
        
        if self.dem_data is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        current_res = self.get_resolution()[0]
        scale_factor = current_res / target_resolution
        
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'cubic':
            order = 3
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        resampled = ndimage.zoom(self.dem_data, scale_factor, order=order)
        
        return resampled
    
    def fill_sinks(self, max_depth: float = 10.0) -> np.ndarray:
        """
        Fill sinks (local depressions) in the DEM.
        
        This is important for hydrological analysis to ensure
        water can flow continuously across the surface.
        
        Args:
            max_depth: Maximum depth of sinks to fill
            
        Returns:
            Filled DEM array
        """
        if self.dem_data is None:
            raise ValueError("DEM not loaded. Call load_dem() first.")
        
        from scipy import ndimage
        
        filled = self.dem_data.copy()
        
        # Simple sink filling using iterative morphological operations
        for _ in range(100):  # Max iterations
            # Find local minima
            local_min = ndimage.minimum_filter(filled, size=3)
            
            # Identify sinks (pixels lower than all neighbors)
            sinks = (filled < local_min + 0.001) & (filled > local_min - max_depth)
            
            if not np.any(sinks):
                break
            
            # Raise sink pixels to the level of their lowest neighbor
            filled[sinks] = local_min[sinks] + 0.001
        
        return filled
