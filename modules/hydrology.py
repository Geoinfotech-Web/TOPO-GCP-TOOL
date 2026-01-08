"""
Hydrology Analysis Module
=========================
Calculate flow direction, flow accumulation, watersheds, and stream networks.
"""

import numpy as np
from rasterio.transform import Affine
from rasterio.crs import CRS
from typing import Tuple, Optional, List, Dict, Any
from scipy import ndimage
import json
from shapely.geometry import LineString, mapping


# D8 flow direction encoding
# Direction values represent the direction of flow:
#  32  64  128
#  16   0   1
#   8   4   2
D8_DIRECTIONS = {
    0: (0, 0),    # No flow (flat or sink)
    1: (0, 1),    # East
    2: (1, 1),    # Southeast
    4: (1, 0),    # South
    8: (1, -1),   # Southwest
    16: (0, -1),  # West
    32: (-1, -1), # Northwest
    64: (-1, 0),  # North
    128: (-1, 1)  # Northeast
}

# Reverse mapping for neighbor indices
NEIGHBOR_OFFSETS = [
    (-1, -1, 32),   # NW
    (-1, 0, 64),    # N
    (-1, 1, 128),   # NE
    (0, -1, 16),    # W
    (0, 1, 1),      # E
    (1, -1, 8),     # SW
    (1, 0, 4),      # S
    (1, 1, 2),      # SE
]


def fill_depressions(dem: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    """
    Fill depressions (sinks) in the DEM to ensure continuous flow.
    
    Uses a simple iterative approach that raises each cell to the
    minimum of its neighbors if it's lower than all neighbors.
    
    Args:
        dem: 2D numpy array of elevation values
        max_iterations: Maximum number of iterations
        
    Returns:
        Filled DEM array
    """
    filled = dem.copy()
    
    for iteration in range(max_iterations):
        changes = 0
        
        # Pad for boundary handling
        padded = np.pad(filled, 1, mode='edge')
        
        for dr, dc, _ in NEIGHBOR_OFFSETS:
            neighbor = padded[1+dr:padded.shape[0]-1+dr, 
                            1+dc:padded.shape[1]-1+dc]
            
            # Find cells that need to be raised
            needs_fill = (filled < neighbor) & ~np.isnan(filled) & ~np.isnan(neighbor)
            
            if np.any(needs_fill):
                # This is a simplified approach - real algorithms are more sophisticated
                pass
        
        # Check for convergence using local minimum filter
        local_min = ndimage.minimum_filter(filled, size=3, mode='constant', cval=np.inf)
        sinks = (filled < local_min) & ~np.isnan(filled)
        
        if not np.any(sinks):
            break
            
        # Raise sinks slightly
        filled[sinks] = local_min[sinks]
        changes = np.sum(sinks)
        
        if changes == 0:
            break
    
    return filled


def calculate_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Calculate D8 flow direction from a DEM.
    
    The D8 algorithm assigns flow to one of 8 neighboring cells
    based on the steepest downhill gradient.
    
    Direction encoding:
      32  64  128
      16   X   1
       8   4   2
    
    Args:
        dem: 2D numpy array of elevation values
        
    Returns:
        2D numpy array of D8 flow direction values
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)
    
    # Pad DEM for edge handling
    padded = np.pad(dem, 1, mode='edge')
    
    # Calculate drop to each neighbor
    # Weights for diagonal vs cardinal directions (1.0 vs sqrt(2))
    weights = [np.sqrt(2), 1.0, np.sqrt(2), 1.0, 1.0, np.sqrt(2), 1.0, np.sqrt(2)]
    
    max_drop = np.zeros((rows, cols))
    
    for i, (dr, dc, direction) in enumerate(NEIGHBOR_OFFSETS):
        neighbor = padded[1+dr:rows+1+dr, 1+dc:cols+1+dc]
        
        # Calculate drop (positive = downhill)
        drop = (dem - neighbor) / weights[i]
        
        # Update flow direction where this is the steepest drop
        steeper = drop > max_drop
        flow_dir[steeper] = direction
        max_drop[steeper] = drop[steeper]
    
    # Handle flat areas and NaN values
    flow_dir[np.isnan(dem)] = 0
    flow_dir[max_drop <= 0] = 0  # Sinks or flat areas
    
    return flow_dir


def calculate_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Calculate flow accumulation from flow direction.
    
    Flow accumulation represents the number of upstream cells
    that flow into each cell.
    
    Args:
        flow_dir: D8 flow direction array
        
    Returns:
        2D numpy array of flow accumulation values
    """
    rows, cols = flow_dir.shape
    flow_acc = np.ones((rows, cols), dtype=np.float32)
    
    # Create a count of how many cells flow into each cell
    inflow_count = np.zeros((rows, cols), dtype=np.int32)
    
    # First pass: count inflows
    for dr, dc, direction in NEIGHBOR_OFFSETS:
        # Find cells whose flow direction points to the center
        # The opposite direction would flow toward us
        opposite = {
            1: 16, 2: 32, 4: 64, 8: 128,
            16: 1, 32: 2, 64: 4, 128: 8
        }
        
        # Shift flow_dir to check which neighbors point to us
        shifted = np.roll(np.roll(flow_dir, -dr, axis=0), -dc, axis=1)
        flows_here = shifted == opposite.get(direction, 0)
        
        # Handle boundaries
        if dr == -1:
            flows_here[-1, :] = False
        elif dr == 1:
            flows_here[0, :] = False
        if dc == -1:
            flows_here[:, -1] = False
        elif dc == 1:
            flows_here[:, 0] = False
            
        inflow_count += flows_here.astype(np.int32)
    
    # Process cells in order from headwaters (no inflow) to outlets
    # Use iterative approach
    processed = np.zeros((rows, cols), dtype=bool)
    remaining_inflows = inflow_count.copy()
    
    for _ in range(rows * cols):
        # Find cells with no remaining inflows
        ready = (remaining_inflows == 0) & ~processed
        
        if not np.any(ready):
            break
        
        # Get coordinates of ready cells
        ready_rows, ready_cols = np.where(ready)
        
        for r, c in zip(ready_rows, ready_cols):
            processed[r, c] = True
            
            # Find where this cell flows to
            direction = flow_dir[r, c]
            if direction == 0:
                continue
            
            dr, dc = D8_DIRECTIONS.get(direction, (0, 0))
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                flow_acc[nr, nc] += flow_acc[r, c]
                remaining_inflows[nr, nc] -= 1
    
    return flow_acc


def delineate_watersheds(flow_dir: np.ndarray, flow_acc: np.ndarray,
                         threshold: float = None,
                         num_watersheds: int = None) -> np.ndarray:
    """
    Delineate watershed basins from flow direction.
    
    Args:
        flow_dir: D8 flow direction array
        flow_acc: Flow accumulation array
        threshold: Minimum flow accumulation for pour points
        num_watersheds: Target number of watersheds
        
    Returns:
        2D numpy array of watershed IDs (0 = no watershed)
    """
    rows, cols = flow_dir.shape
    
    # Find pour points (outlets) - typically high accumulation points at edges
    # or local accumulation maxima
    
    if threshold is None:
        # Use top percentile of flow accumulation
        threshold = np.nanpercentile(flow_acc, 99)
    
    # Find potential pour points
    pour_points = flow_acc >= threshold
    
    # Also consider edge cells with high accumulation
    edge_mask = np.zeros_like(pour_points)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    
    pour_points = pour_points | (edge_mask & (flow_acc > np.nanmedian(flow_acc)))
    
    # If num_watersheds specified, select top N pour points
    if num_watersheds is not None:
        pour_point_acc = flow_acc.copy()
        pour_point_acc[~pour_points] = 0
        
        # Get sorted indices
        flat_indices = np.argsort(pour_point_acc.flatten())[::-1]
        selected = flat_indices[:num_watersheds]
        
        new_pour_points = np.zeros_like(pour_points)
        for idx in selected:
            r, c = np.unravel_index(idx, pour_points.shape)
            new_pour_points[r, c] = True
        pour_points = new_pour_points
    
    # Assign watershed IDs
    watersheds = np.zeros((rows, cols), dtype=np.int32)
    
    # Label pour points
    pour_rows, pour_cols = np.where(pour_points)
    for i, (r, c) in enumerate(zip(pour_rows, pour_cols), 1):
        watersheds[r, c] = i
    
    # Trace upstream from each cell to assign to watershed
    # This is done iteratively
    for _ in range(max(rows, cols)):
        changed = False
        
        for dr, dc, direction in NEIGHBOR_OFFSETS:
            # Shift watersheds to see what downstream cell's watershed is
            # If a cell flows to a cell with a watershed ID, inherit it
            
            # Find cells that flow in this direction
            flows_this_way = flow_dir == direction
            
            # Get the watershed of the downstream cell
            downstream_ws = np.roll(np.roll(watersheds, -dr, axis=0), -dc, axis=1)
            
            # Handle boundaries
            if dr == -1:
                downstream_ws[-1, :] = 0
            elif dr == 1:
                downstream_ws[0, :] = 0
            if dc == -1:
                downstream_ws[:, -1] = 0
            elif dc == 1:
                downstream_ws[:, 0] = 0
            
            # Assign watershed to cells that flow to a labeled cell
            unassigned = (watersheds == 0) & flows_this_way & (downstream_ws > 0)
            
            if np.any(unassigned):
                watersheds[unassigned] = downstream_ws[unassigned]
                changed = True
        
        if not changed:
            break
    
    return watersheds


def extract_streams(flow_acc: np.ndarray, flow_dir: np.ndarray,
                    transform: Affine, crs: CRS,
                    threshold: float, output_path: str) -> int:
    """
    Extract stream network from flow accumulation.
    
    Streams are defined as cells with flow accumulation above
    the specified threshold.
    
    Args:
        flow_acc: Flow accumulation array
        flow_dir: Flow direction array
        transform: Affine transformation
        crs: Coordinate reference system
        threshold: Minimum accumulation to define a stream
        output_path: Output file path
        
    Returns:
        Number of stream segments
    """
    rows, cols = flow_acc.shape
    
    # Identify stream cells
    stream_cells = flow_acc >= threshold
    
    # Label connected stream segments
    labeled, num_segments = ndimage.label(stream_cells)
    
    # Trace each segment to create line features
    features = []
    
    for segment_id in range(1, num_segments + 1):
        segment_mask = labeled == segment_id
        segment_rows, segment_cols = np.where(segment_mask)
        
        if len(segment_rows) < 2:
            continue
        
        # Find the upstream end (lowest accumulation in segment)
        segment_acc = flow_acc[segment_mask]
        upstream_idx = np.argmin(segment_acc)
        
        # Trace downstream from upstream end
        r, c = segment_rows[upstream_idx], segment_cols[upstream_idx]
        coords = []
        visited = set()
        
        while True:
            if (r, c) in visited:
                break
            visited.add((r, c))
            
            # Convert to geographic coordinates
            x = transform.c + c * transform.a + r * transform.b
            y = transform.f + c * transform.d + r * transform.e
            coords.append((x, y))
            
            # Get flow direction
            direction = flow_dir[r, c]
            if direction == 0:
                break
            
            # Move to next cell
            dr, dc = D8_DIRECTIONS.get(direction, (0, 0))
            nr, nc = r + dr, c + dc
            
            # Check bounds and if still in stream
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if not stream_cells[nr, nc]:
                # Add final point and stop
                x = transform.c + nc * transform.a + nr * transform.b
                y = transform.f + nc * transform.d + nr * transform.e
                coords.append((x, y))
                break
            
            r, c = nr, nc
        
        if len(coords) >= 2:
            try:
                line = LineString(coords)
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "segment_id": int(segment_id),
                        "order": calculate_stream_order(segment_id, labeled, flow_dir),
                        "length": line.length
                    },
                    "geometry": mapping(line)
                }
                features.append(feature)
            except Exception:
                continue
    
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


def calculate_stream_order(segment_id: int, labeled: np.ndarray, 
                           flow_dir: np.ndarray) -> int:
    """
    Calculate Strahler stream order for a segment.
    
    This is a simplified calculation - full Strahler ordering
    requires complete network topology.
    
    Args:
        segment_id: ID of the stream segment
        labeled: Labeled stream array
        flow_dir: Flow direction array
        
    Returns:
        Stream order (1 = headwater)
    """
    # Count tributaries flowing into this segment
    segment_mask = labeled == segment_id
    
    # Find cells at the upstream edge of segment
    eroded = ndimage.binary_erosion(segment_mask)
    upstream_edge = segment_mask & ~eroded
    
    # Count unique segments flowing into this one
    tributary_count = 0
    for dr, dc, direction in NEIGHBOR_OFFSETS:
        opposite = {
            1: 16, 2: 32, 4: 64, 8: 128,
            16: 1, 32: 2, 64: 4, 128: 8
        }
        
        # Check if neighbor is different segment flowing here
        shifted_labeled = np.roll(np.roll(labeled, -dr, axis=0), -dc, axis=1)
        shifted_flow = np.roll(np.roll(flow_dir, -dr, axis=0), -dc, axis=1)
        
        flows_in = (shifted_flow == opposite.get(direction, 0)) & \
                   (shifted_labeled != segment_id) & \
                   (shifted_labeled > 0) & \
                   segment_mask
        
        tributary_count += np.sum(flows_in)
    
    # Simple order calculation
    if tributary_count == 0:
        return 1
    elif tributary_count == 1:
        return 2
    else:
        return min(tributary_count, 5)


def calculate_twi(dem: np.ndarray, flow_acc: np.ndarray,
                  transform: Affine) -> np.ndarray:
    """
    Calculate Topographic Wetness Index (TWI).
    
    TWI = ln(a / tan(β))
    where a = specific catchment area and β = slope
    
    Higher TWI values indicate areas likely to be wet/saturated.
    
    Args:
        dem: Elevation array
        flow_acc: Flow accumulation array
        transform: Affine transformation
        
    Returns:
        2D array of TWI values
    """
    from modules.slope_analysis import calculate_slope
    
    # Calculate slope in radians
    slope = calculate_slope(dem, transform, units='radians')
    
    # Avoid division by zero for flat areas
    slope = np.maximum(slope, 0.001)
    
    # Calculate specific catchment area (flow_acc * cell_size / cell_size = flow_acc)
    cell_size = abs(transform.a)
    specific_catchment = flow_acc * cell_size
    
    # Calculate TWI
    twi = np.log(specific_catchment / np.tan(slope))
    
    # Handle invalid values
    twi[np.isnan(dem)] = np.nan
    twi[np.isinf(twi)] = np.nan
    
    return twi.astype(np.float32)
