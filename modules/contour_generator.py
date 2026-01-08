"""
Contour Generator Module
========================
Generate contour lines from Digital Elevation Models.
"""

import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
from typing import List, Tuple, Optional, Dict, Any
import json
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge
import warnings


def generate_contours(dem: np.ndarray, transform: Affine, crs: CRS,
                      interval: float, output_path: str,
                      base: float = 0.0, 
                      smooth: bool = True,
                      min_length: float = 10.0) -> int:
    """
    Generate contour lines from a DEM and save to GeoJSON.
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        crs: Coordinate reference system
        interval: Contour interval in map units (e.g., meters)
        output_path: Path for output GeoJSON file
        base: Base elevation to start from
        smooth: Whether to smooth the contour lines
        min_length: Minimum contour length to include
        
    Returns:
        Number of contour lines generated
    """
    # Handle NaN values
    dem_filled = np.nan_to_num(dem, nan=np.nanmin(dem) - 1)
    
    # Calculate elevation range
    min_elev = np.nanmin(dem)
    max_elev = np.nanmax(dem)
    
    # Generate contour levels
    start_level = np.ceil((min_elev - base) / interval) * interval + base
    end_level = np.floor((max_elev - base) / interval) * interval + base
    levels = np.arange(start_level, end_level + interval, interval)
    
    # Use matplotlib to generate contours
    import matplotlib.pyplot as plt
    from matplotlib.contour import ContourGenerator
    
    # Create coordinate arrays
    rows, cols = dem.shape
    x = np.arange(cols)
    y = np.arange(rows)
    
    # Generate contours using matplotlib
    fig, ax = plt.subplots()
    contour_set = ax.contour(x, y, dem_filled, levels=levels)
    plt.close(fig)
    
    # Extract contour lines and convert to geographic coordinates
    features = []
    
    for level_idx, level in enumerate(contour_set.levels):
        # Get paths for this level
        paths = contour_set.collections[level_idx].get_paths()
        
        for path in paths:
            vertices = path.vertices
            
            if len(vertices) < 2:
                continue
            
            # Convert pixel coordinates to geographic coordinates
            geo_coords = []
            for px, py in vertices:
                # Apply affine transformation
                gx = transform.c + px * transform.a + py * transform.b
                gy = transform.f + px * transform.d + py * transform.e
                geo_coords.append((gx, gy))
            
            # Create LineString
            try:
                line = LineString(geo_coords)
                
                # Check minimum length
                if line.length < min_length:
                    continue
                
                # Smooth if requested
                if smooth:
                    line = smooth_line(line, tolerance=transform.a * 0.5)
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "elevation": float(level),
                        "contour_type": "index" if level % (interval * 5) == 0 else "intermediate"
                    },
                    "geometry": mapping(line)
                }
                features.append(feature)
                
            except Exception as e:
                warnings.warn(f"Error processing contour at level {level}: {e}")
                continue
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": str(crs) if crs else "urn:ogc:def:crs:EPSG::4326"
            }
        },
        "features": features
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    return len(features)


def smooth_line(line: LineString, tolerance: float = 1.0) -> LineString:
    """
    Smooth a line using the Ramer-Douglas-Peucker algorithm
    followed by Chaikin's corner cutting.
    
    Args:
        line: Input LineString
        tolerance: Simplification tolerance
        
    Returns:
        Smoothed LineString
    """
    # First simplify to reduce noise
    simplified = line.simplify(tolerance, preserve_topology=True)
    
    if len(simplified.coords) < 3:
        return simplified
    
    # Apply Chaikin's corner cutting algorithm
    coords = list(simplified.coords)
    
    for _ in range(2):  # Apply twice for smoother result
        new_coords = [coords[0]]  # Keep first point
        
        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            
            # Create two new points at 1/4 and 3/4 positions
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            
            new_coords.extend([q, r])
        
        new_coords.append(coords[-1])  # Keep last point
        coords = new_coords
    
    return LineString(coords)


def generate_contours_labeled(dem: np.ndarray, transform: Affine, crs: CRS,
                               interval: float, output_path: str,
                               label_interval: int = 5) -> int:
    """
    Generate contours with labels for index contours.
    
    This creates contours with properties indicating whether
    they are index contours (every nth contour) which should
    be drawn thicker and labeled.
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        crs: Coordinate reference system
        interval: Contour interval
        output_path: Output file path
        label_interval: Every nth contour is an index contour
        
    Returns:
        Number of contours generated
    """
    return generate_contours(dem, transform, crs, interval, output_path,
                            smooth=True, min_length=10.0)


def calculate_contour_statistics(dem: np.ndarray, interval: float) -> Dict[str, Any]:
    """
    Calculate statistics about contour distribution.
    
    Args:
        dem: 2D numpy array of elevation values
        interval: Contour interval
        
    Returns:
        Dictionary with contour statistics
    """
    min_elev = np.nanmin(dem)
    max_elev = np.nanmax(dem)
    elev_range = max_elev - min_elev
    
    num_contours = int(elev_range / interval) + 1
    
    # Calculate histogram of elevations
    valid_dem = dem[~np.isnan(dem)]
    hist, bin_edges = np.histogram(valid_dem, bins=num_contours)
    
    return {
        'min_elevation': float(min_elev),
        'max_elevation': float(max_elev),
        'elevation_range': float(elev_range),
        'interval': float(interval),
        'num_contours': num_contours,
        'elevation_distribution': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    }


def contours_to_shapefile(geojson_path: str, shapefile_path: str, crs: CRS) -> str:
    """
    Convert contour GeoJSON to Shapefile format.
    
    Args:
        geojson_path: Path to input GeoJSON
        shapefile_path: Path for output Shapefile
        crs: Coordinate reference system
        
    Returns:
        Path to created Shapefile
    """
    import fiona
    from fiona.crs import from_epsg
    
    # Read GeoJSON
    with open(geojson_path, 'r') as f:
        geojson = json.load(f)
    
    # Define schema
    schema = {
        'geometry': 'LineString',
        'properties': {
            'elevation': 'float',
            'contour_type': 'str'
        }
    }
    
    # Get CRS for Fiona
    if crs:
        try:
            fiona_crs = crs.to_dict()
        except:
            fiona_crs = from_epsg(4326)
    else:
        fiona_crs = from_epsg(4326)
    
    # Write Shapefile
    with fiona.open(shapefile_path, 'w', driver='ESRI Shapefile',
                    schema=schema, crs=fiona_crs) as shp:
        for feature in geojson['features']:
            shp.write(feature)
    
    return shapefile_path


def generate_spot_heights(dem: np.ndarray, transform: Affine, crs: CRS,
                          output_path: str, strategy: str = 'peaks',
                          min_spacing: float = 50.0) -> int:
    """
    Generate spot height points for map labeling.
    
    Spot heights are points placed at significant elevation locations
    like peaks, saddles, and breaks in slope.
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation for the raster
        crs: Coordinate reference system
        output_path: Output file path
        strategy: 'peaks', 'grid', or 'significant'
        min_spacing: Minimum spacing between spot heights
        
    Returns:
        Number of spot heights generated
    """
    from scipy import ndimage
    
    features = []
    cell_size = abs(transform.a)
    spacing_pixels = int(min_spacing / cell_size)
    
    if strategy == 'peaks':
        # Find local maxima
        local_max = ndimage.maximum_filter(dem, size=spacing_pixels)
        peaks = (dem == local_max) & ~np.isnan(dem)
        
        # Get peak locations
        rows, cols = np.where(peaks)
        
    elif strategy == 'grid':
        # Regular grid of points
        rows = np.arange(spacing_pixels // 2, dem.shape[0], spacing_pixels)
        cols = np.arange(spacing_pixels // 2, dem.shape[1], spacing_pixels)
        rows, cols = np.meshgrid(rows, cols, indexing='ij')
        rows = rows.flatten()
        cols = cols.flatten()
        
    else:  # significant
        # Find peaks, saddles, and slope breaks
        local_max = ndimage.maximum_filter(dem, size=spacing_pixels)
        local_min = ndimage.minimum_filter(dem, size=spacing_pixels)
        
        # Peaks
        peaks = dem == local_max
        
        # Saddles (points that are both local max in one direction and min in another)
        saddles = (dem == local_max) | (dem == local_min)
        
        combined = (peaks | saddles) & ~np.isnan(dem)
        rows, cols = np.where(combined)
    
    # Convert to features
    for row, col in zip(rows, cols):
        if np.isnan(dem[row, col]):
            continue
            
        # Convert to geographic coordinates
        x = transform.c + col * transform.a + row * transform.b
        y = transform.f + col * transform.d + row * transform.e
        
        feature = {
            "type": "Feature",
            "properties": {
                "elevation": float(dem[row, col]),
                "type": "spot_height"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [x, y]
            }
        }
        features.append(feature)
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": str(crs) if crs else "urn:ogc:def:crs:EPSG::4326"
            }
        },
        "features": features
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    return len(features)
