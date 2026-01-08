"""
GCP Generator Module
====================
Generate Ground Control Point markers for topographic surveys.
"""

import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
from typing import List, Dict, Any, Optional, Tuple
from scipy import ndimage


def generate_gcp_markers(dem: np.ndarray, transform: Affine, crs: CRS,
                         spacing: float = 100.0,
                         strategy: str = 'grid_pattern',
                         edge_buffer: float = 10.0,
                         max_slope: float = 30.0) -> List[Dict[str, Any]]:
    """
    Generate Ground Control Point markers for field placement.
    
    GCPs are used to georeference drone imagery and validate
    accuracy of the resulting DEM and orthophotos.
    
    Args:
        dem: 2D numpy array of elevation values
        transform: Affine transformation
        crs: Coordinate reference system
        spacing: Desired spacing between GCPs in map units
        strategy: Placement strategy - 'grid_pattern', 'terrain_adaptive', or 'edge___interior'
        edge_buffer: Buffer from edges in map units
        max_slope: Maximum slope (degrees) for GCP placement
        
    Returns:
        List of GCP dictionaries with coordinates and metadata
    """
    rows, cols = dem.shape
    cell_size = abs(transform.a)
    
    # Calculate spacing in pixels
    spacing_pixels = max(int(spacing / cell_size), 1)
    buffer_pixels = max(int(edge_buffer / cell_size), 1)
    
    # Calculate slope to avoid placing GCPs on steep terrain
    from modules.slope_analysis import calculate_slope
    slope = calculate_slope(dem, transform, units='degrees')
    
    # Create valid placement mask
    valid_mask = ~np.isnan(dem) & (slope <= max_slope)
    
    # Apply edge buffer
    if buffer_pixels < rows // 2 and buffer_pixels < cols // 2:
        valid_mask[:buffer_pixels, :] = False
        valid_mask[-buffer_pixels:, :] = False
        valid_mask[:, :buffer_pixels] = False
        valid_mask[:, -buffer_pixels:] = False
    
    gcps = []
    
    if strategy == 'grid_pattern':
        gcps = _generate_grid_gcps(dem, transform, crs, valid_mask, 
                                   spacing_pixels, buffer_pixels, slope)
    
    elif strategy == 'terrain_adaptive':
        gcps = _generate_terrain_adaptive_gcps(dem, transform, crs, valid_mask,
                                                spacing_pixels, buffer_pixels, slope)
    
    elif strategy == 'edge___interior':
        gcps = _generate_edge_interior_gcps(dem, transform, crs, valid_mask,
                                             spacing_pixels, buffer_pixels, slope)
    
    else:
        # Default to grid pattern
        gcps = _generate_grid_gcps(dem, transform, crs, valid_mask,
                                   spacing_pixels, buffer_pixels, slope)
    
    # Add sequential IDs and format
    for i, gcp in enumerate(gcps, 1):
        gcp['id'] = f"GCP_{i:03d}"
    
    return gcps


def _generate_grid_gcps(dem: np.ndarray, transform: Affine, crs: CRS,
                        valid_mask: np.ndarray, spacing: int, 
                        buffer: int, slope: np.ndarray) -> List[Dict]:
    """Generate GCPs in a regular grid pattern."""
    rows, cols = dem.shape
    gcps = []
    
    # Ensure we get at least some points
    spacing = max(spacing, 1)
    
    # Generate grid points
    for row in range(buffer, rows - buffer, spacing):
        for col in range(buffer, cols - buffer, spacing):
            if valid_mask[row, col]:
                gcp = _create_gcp(row, col, dem, transform, crs, slope, 'grid')
                if gcp:
                    gcps.append(gcp)
            else:
                # Find nearest valid cell
                nr, nc = _find_nearest_valid(row, col, valid_mask, spacing // 2)
                if nr is not None:
                    gcp = _create_gcp(nr, nc, dem, transform, crs, slope, 'grid')
                    if gcp:
                        gcps.append(gcp)
    
    return gcps


def _generate_terrain_adaptive_gcps(dem: np.ndarray, transform: Affine, crs: CRS,
                                     valid_mask: np.ndarray, spacing: int,
                                     buffer: int, slope: np.ndarray) -> List[Dict]:
    """
    Generate GCPs that adapt to terrain features.
    
    Places points at:
    - Terrain breaks (changes in slope)
    - Ridge lines and valley bottoms
    - Flat areas (preferred for accuracy)
    """
    rows, cols = dem.shape
    gcps = []
    
    # Find terrain features
    
    # 1. Flat areas (low slope) - preferred locations
    flat_areas = slope < 5.0
    
    # 2. Ridge lines and valleys using curvature
    from modules.slope_analysis import calculate_curvature
    curvature = calculate_curvature(dem, transform, 'total')
    
    ridges = curvature > np.nanpercentile(curvature[~np.isnan(curvature)], 90)
    valleys = curvature < np.nanpercentile(curvature[~np.isnan(curvature)], 10)
    
    # Combine features
    feature_points = (flat_areas | ridges | valleys) & valid_mask
    
    # Sample points from features
    feature_rows, feature_cols = np.where(feature_points)
    
    if len(feature_rows) == 0:
        # Fall back to grid
        return _generate_grid_gcps(dem, transform, crs, valid_mask, 
                                   spacing, buffer, slope)
    
    # Use k-means-like approach to select well-distributed points
    selected_indices = _select_distributed_points(
        feature_rows, feature_cols, 
        target_spacing=spacing,
        max_points=100
    )
    
    for idx in selected_indices:
        row, col = feature_rows[idx], feature_cols[idx]
        
        # Determine point type
        if flat_areas[row, col]:
            point_type = 'flat_area'
        elif ridges[row, col]:
            point_type = 'ridge'
        else:
            point_type = 'valley'
        
        gcp = _create_gcp(row, col, dem, transform, crs, slope, point_type)
        if gcp:
            gcps.append(gcp)
    
    return gcps


def _generate_edge_interior_gcps(dem: np.ndarray, transform: Affine, crs: CRS,
                                  valid_mask: np.ndarray, spacing: int,
                                  buffer: int, slope: np.ndarray) -> List[Dict]:
    """
    Generate GCPs along edges and in the interior.
    
    This strategy ensures good coverage at the edges of the survey area
    (critical for geometric accuracy) plus interior points for tie-points.
    """
    rows, cols = dem.shape
    gcps = []
    
    # Edge points - along the perimeter
    edge_spacing = spacing
    
    # Top edge
    for col in range(buffer, cols - buffer, edge_spacing):
        row = _find_valid_in_range(buffer, min(buffer + spacing // 2, rows), col, valid_mask)
        if row is not None:
            gcp = _create_gcp(row, col, dem, transform, crs, slope, 'edge_top')
            if gcp:
                gcps.append(gcp)
    
    # Bottom edge
    for col in range(buffer, cols - buffer, edge_spacing):
        row = _find_valid_in_range(max(0, rows - buffer - spacing // 2), rows - buffer, col, valid_mask)
        if row is not None:
            gcp = _create_gcp(row, col, dem, transform, crs, slope, 'edge_bottom')
            if gcp:
                gcps.append(gcp)
    
    # Left edge
    for row in range(buffer + edge_spacing, rows - buffer - edge_spacing, edge_spacing):
        col = _find_valid_in_range_col(buffer, min(buffer + spacing // 2, cols), row, valid_mask)
        if col is not None:
            gcp = _create_gcp(row, col, dem, transform, crs, slope, 'edge_left')
            if gcp:
                gcps.append(gcp)
    
    # Right edge
    for row in range(buffer + edge_spacing, rows - buffer - edge_spacing, edge_spacing):
        col = _find_valid_in_range_col(max(0, cols - buffer - spacing // 2), cols - buffer, row, valid_mask)
        if col is not None:
            gcp = _create_gcp(row, col, dem, transform, crs, slope, 'edge_right')
            if gcp:
                gcps.append(gcp)
    
    # Interior points - sparser grid
    interior_spacing = int(spacing * 1.5)
    for row in range(buffer + interior_spacing, rows - buffer - interior_spacing, interior_spacing):
        for col in range(buffer + interior_spacing, cols - buffer - interior_spacing, interior_spacing):
            if valid_mask[row, col]:
                gcp = _create_gcp(row, col, dem, transform, crs, slope, 'interior')
                if gcp:
                    gcps.append(gcp)
    
    return gcps


def _find_valid_in_range(row_start: int, row_end: int, col: int, 
                         valid_mask: np.ndarray) -> Optional[int]:
    """Find a valid row within a range for a given column."""
    rows = valid_mask.shape[0]
    cols = valid_mask.shape[1]
    
    if col < 0 or col >= cols:
        return None
    
    for row in range(max(0, row_start), min(rows, row_end)):
        if valid_mask[row, col]:
            return row
    return None


def _find_valid_in_range_col(col_start: int, col_end: int, row: int,
                             valid_mask: np.ndarray) -> Optional[int]:
    """Find a valid column within a range for a given row."""
    rows = valid_mask.shape[0]
    cols = valid_mask.shape[1]
    
    if row < 0 or row >= rows:
        return None
        
    for col in range(max(0, col_start), min(cols, col_end)):
        if valid_mask[row, col]:
            return col
    return None


def _find_nearest_valid(row: int, col: int, valid_mask: np.ndarray,
                        search_radius: int) -> Tuple[Optional[int], Optional[int]]:
    """Find the nearest valid cell within a search radius."""
    rows, cols = valid_mask.shape
    search_radius = max(search_radius, 1)
    
    for r in range(search_radius):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    valid_mask[nr, nc]):
                    return nr, nc
    
    return None, None


def _create_gcp(row: int, col: int, dem: np.ndarray, transform: Affine,
                crs: CRS, slope: np.ndarray, point_type: str) -> Optional[Dict]:
    """Create a GCP dictionary from pixel coordinates."""
    if row is None or col is None:
        return None
    
    rows, cols = dem.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None
    
    # Get elevation
    elevation = dem[row, col]
    if np.isnan(elevation):
        return None
    
    # Convert to geographic coordinates
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    
    # Get slope at point
    point_slope = slope[row, col] if not np.isnan(slope[row, col]) else 0.0
    
    return {
        'x': float(x),
        'y': float(y),
        'elevation': float(elevation),
        'slope': float(point_slope),
        'type': point_type,
        'row': int(row),
        'col': int(col)
    }


def _select_distributed_points(rows: np.ndarray, cols: np.ndarray,
                                target_spacing: int, max_points: int) -> List[int]:
    """
    Select well-distributed points from a set of candidates.
    
    Uses a greedy approach similar to Poisson disk sampling.
    """
    if len(rows) <= max_points:
        return list(range(len(rows)))
    
    # Start with first point
    selected = [0]
    
    while len(selected) < max_points:
        # Find point farthest from all selected points
        max_min_dist = 0
        best_idx = None
        
        for i in range(len(rows)):
            if i in selected:
                continue
            
            # Calculate minimum distance to selected points
            min_dist = float('inf')
            for sel_idx in selected:
                dist = np.sqrt((rows[i] - rows[sel_idx])**2 + 
                              (cols[i] - cols[sel_idx])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        
        if best_idx is None or max_min_dist < target_spacing * 0.5:
            break
        
        selected.append(best_idx)
    
    return selected


def optimize_gcp_distribution(gcps: List[Dict], min_spacing: float,
                              bounds: Tuple[float, float, float, float]) -> List[Dict]:
    """
    Optimize GCP distribution by removing clustered points.
    
    Args:
        gcps: List of GCP dictionaries
        min_spacing: Minimum allowed spacing between GCPs
        bounds: (min_x, min_y, max_x, max_y) survey bounds
        
    Returns:
        Optimized list of GCPs
    """
    if len(gcps) <= 4:
        return gcps
    
    optimized = []
    
    for gcp in gcps:
        # Check if too close to existing optimized points
        too_close = False
        for existing in optimized:
            dist = np.sqrt((gcp['x'] - existing['x'])**2 + 
                          (gcp['y'] - existing['y'])**2)
            if dist < min_spacing:
                too_close = True
                break
        
        if not too_close:
            optimized.append(gcp)
    
    return optimized


def calculate_gcp_accuracy_estimate(gcps: List[Dict], dem_resolution: float,
                                     survey_area: float) -> Dict[str, Any]:
    """
    Estimate expected accuracy based on GCP distribution.
    
    Args:
        gcps: List of GCP dictionaries
        dem_resolution: DEM cell size
        survey_area: Total survey area
        
    Returns:
        Dictionary with accuracy estimates
    """
    n_gcps = len(gcps)
    
    if n_gcps < 3:
        return {
            'warning': 'Minimum 3 GCPs required for georeferencing',
            'estimated_accuracy': None
        }
    
    # Calculate mean distance between GCPs
    total_dist = 0
    count = 0
    for i, gcp1 in enumerate(gcps):
        for gcp2 in gcps[i+1:]:
            dist = np.sqrt((gcp1['x'] - gcp2['x'])**2 + 
                          (gcp1['y'] - gcp2['y'])**2)
            total_dist += dist
            count += 1
    
    mean_spacing = total_dist / count if count > 0 else 0
    
    # Estimate horizontal accuracy (rough empirical formula)
    estimated_h_accuracy = dem_resolution * (1 + 500 / max(mean_spacing, 1))
    
    # Estimate vertical accuracy (typically 1.5-2x horizontal)
    estimated_v_accuracy = estimated_h_accuracy * 1.5
    
    return {
        'num_gcps': n_gcps,
        'mean_spacing': float(mean_spacing),
        'gcp_density': n_gcps / max((survey_area / 10000), 1),
        'estimated_horizontal_accuracy': float(min(estimated_h_accuracy, dem_resolution * 3)),
        'estimated_vertical_accuracy': float(min(estimated_v_accuracy, dem_resolution * 4.5)),
        'distribution_quality': _assess_distribution_quality(gcps)
    }


def _assess_distribution_quality(gcps: List[Dict]) -> str:
    """Assess the quality of GCP distribution."""
    if len(gcps) < 4:
        return 'insufficient'
    
    # Check for edge coverage
    edge_count = sum(1 for gcp in gcps if gcp.get('type', '').startswith('edge'))
    
    if edge_count >= 4 and len(gcps) >= 9:
        return 'excellent'
    elif edge_count >= 2 and len(gcps) >= 6:
        return 'good'
    elif len(gcps) >= 4:
        return 'adequate'
    else:
        return 'poor'
