"""
Exporters Module
================
Export GCP data to various formats: CSV, KML, Shapefile, DXF.
"""

import csv
import json
from typing import List, Dict, Any, Optional
from rasterio.crs import CRS
import os


def export_to_csv(gcps: List[Dict], output_path: str,
                  include_headers: bool = True) -> str:
    """
    Export GCPs to CSV format.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output CSV file
        include_headers: Whether to include column headers
        
    Returns:
        Path to created file
    """
    if not gcps:
        # Create empty file with headers
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'x', 'y', 'elevation', 'slope', 'type'])
        return output_path
    
    fieldnames = ['id', 'x', 'y', 'elevation', 'slope', 'type']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        
        if include_headers:
            writer.writeheader()
        
        for gcp in gcps:
            row = {
                'id': gcp.get('id', ''),
                'x': f"{gcp['x']:.6f}",
                'y': f"{gcp['y']:.6f}",
                'elevation': f"{gcp['elevation']:.3f}",
                'slope': f"{gcp.get('slope', 0):.2f}",
                'type': gcp.get('type', 'unknown')
            }
            writer.writerow(row)
    
    return output_path


def export_to_kml(gcps: List[Dict], output_path: str,
                  name: str = "GCP Markers",
                  description: str = "Ground Control Points for Survey") -> str:
    """
    Export GCPs to KML format for Google Earth.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output KML file
        name: Name for the KML document
        description: Description for the KML document
        
    Returns:
        Path to created file
    """
    # KML header
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>{name}</name>
    <description>{description}</description>
    
    <!-- GCP Marker Style -->
    <Style id="gcpStyle">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.2</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>
            </Icon>
        </IconStyle>
        <LabelStyle>
            <color>ffffffff</color>
            <scale>0.8</scale>
        </LabelStyle>
    </Style>
    
    <!-- Edge GCP Style -->
    <Style id="edgeGcpStyle">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>1.0</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
            </Icon>
        </IconStyle>
        <LabelStyle>
            <color>ffffffff</color>
            <scale>0.7</scale>
        </LabelStyle>
    </Style>
    
    <Folder>
        <name>Ground Control Points</name>
'''
    
    # Add placemarks for each GCP
    for gcp in gcps:
        gcp_id = gcp.get('id', 'GCP')
        x = gcp['x']
        y = gcp['y']
        elevation = gcp['elevation']
        slope = gcp.get('slope', 0)
        gcp_type = gcp.get('type', 'unknown')
        
        # Choose style based on type
        style = 'edgeGcpStyle' if 'edge' in gcp_type else 'gcpStyle'
        
        kml += f'''
        <Placemark>
            <name>{gcp_id}</name>
            <description>
                <![CDATA[
                    <b>Coordinates:</b><br/>
                    X: {x:.6f}<br/>
                    Y: {y:.6f}<br/>
                    Elevation: {elevation:.3f} m<br/>
                    Slope: {slope:.2f}°<br/>
                    Type: {gcp_type}
                ]]>
            </description>
            <styleUrl>#{style}</styleUrl>
            <Point>
                <altitudeMode>absolute</altitudeMode>
                <coordinates>{x},{y},{elevation}</coordinates>
            </Point>
        </Placemark>
'''
    
    # KML footer
    kml += '''
    </Folder>
</Document>
</kml>'''
    
    with open(output_path, 'w') as f:
        f.write(kml)
    
    return output_path


def export_to_shapefile(gcps: List[Dict], output_path: str,
                        crs: Optional[CRS] = None) -> str:
    """
    Export GCPs to ESRI Shapefile format.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output Shapefile (without extension)
        crs: Coordinate reference system
        
    Returns:
        Path to created Shapefile
    """
    try:
        import fiona
        from fiona.crs import from_epsg
        from shapely.geometry import Point, mapping
        
        # Define schema
        schema = {
            'geometry': 'Point',
            'properties': {
                'id': 'str',
                'elevation': 'float',
                'slope': 'float',
                'type': 'str'
            }
        }
        
        # Get CRS
        if crs:
            try:
                fiona_crs = crs.to_dict()
            except:
                fiona_crs = from_epsg(4326)
        else:
            fiona_crs = from_epsg(4326)
        
        # Ensure output has .shp extension
        if not output_path.endswith('.shp'):
            output_path = output_path + '.shp'
        
        # Write shapefile
        with fiona.open(output_path, 'w', driver='ESRI Shapefile',
                        schema=schema, crs=fiona_crs) as shp:
            for gcp in gcps:
                point = Point(gcp['x'], gcp['y'])
                
                feature = {
                    'geometry': mapping(point),
                    'properties': {
                        'id': gcp.get('id', ''),
                        'elevation': float(gcp['elevation']),
                        'slope': float(gcp.get('slope', 0)),
                        'type': gcp.get('type', 'unknown')
                    }
                }
                shp.write(feature)
        
        return output_path
        
    except ImportError:
        # Fallback: create a simple GeoJSON that can be converted
        geojson_path = output_path.replace('.shp', '.geojson')
        return export_to_geojson(gcps, geojson_path, crs)


def export_to_geojson(gcps: List[Dict], output_path: str,
                      crs: Optional[CRS] = None) -> str:
    """
    Export GCPs to GeoJSON format.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output GeoJSON file
        crs: Coordinate reference system
        
    Returns:
        Path to created file
    """
    features = []
    
    for gcp in gcps:
        feature = {
            "type": "Feature",
            "properties": {
                "id": gcp.get('id', ''),
                "elevation": gcp['elevation'],
                "slope": gcp.get('slope', 0),
                "type": gcp.get('type', 'unknown')
            },
            "geometry": {
                "type": "Point",
                "coordinates": [gcp['x'], gcp['y'], gcp['elevation']]
            }
        }
        features.append(feature)
    
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
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    return output_path


def export_to_dxf(gcps: List[Dict], output_path: str) -> str:
    """
    Export GCPs to DXF format for CAD software.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output DXF file
        
    Returns:
        Path to created file
    """
    try:
        import ezdxf
        
        # Create new DXF document
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Add layer for GCPs
        doc.layers.add('GCP_POINTS', color=1)  # Red
        doc.layers.add('GCP_LABELS', color=7)  # White
        doc.layers.add('GCP_CIRCLES', color=3)  # Green
        
        for gcp in gcps:
            x = gcp['x']
            y = gcp['y']
            z = gcp['elevation']
            gcp_id = gcp.get('id', 'GCP')
            
            # Add point
            msp.add_point((x, y, z), dxfattribs={'layer': 'GCP_POINTS'})
            
            # Add circle marker
            msp.add_circle((x, y, z), radius=2.0, 
                          dxfattribs={'layer': 'GCP_CIRCLES'})
            
            # Add cross marker
            cross_size = 3.0
            msp.add_line((x - cross_size, y, z), (x + cross_size, y, z),
                        dxfattribs={'layer': 'GCP_CIRCLES'})
            msp.add_line((x, y - cross_size, z), (x, y + cross_size, z),
                        dxfattribs={'layer': 'GCP_CIRCLES'})
            
            # Add label
            label_text = f"{gcp_id}\nE: {z:.2f}m"
            msp.add_text(label_text, 
                        dxfattribs={
                            'layer': 'GCP_LABELS',
                            'height': 1.5
                        }).set_pos((x + 4, y + 2, z))
        
        # Save file
        doc.saveas(output_path)
        return output_path
        
    except ImportError:
        # Fallback: Create a simple ASCII DXF manually
        return _create_simple_dxf(gcps, output_path)


def _create_simple_dxf(gcps: List[Dict], output_path: str) -> str:
    """
    Create a simple DXF file without external dependencies.
    
    This creates a basic DXF with points that can be read by most CAD software.
    """
    dxf_content = '''0
SECTION
2
HEADER
0
ENDSEC
0
SECTION
2
ENTITIES
'''
    
    for gcp in gcps:
        x = gcp['x']
        y = gcp['y']
        z = gcp['elevation']
        
        # Add POINT entity
        dxf_content += f'''0
POINT
8
GCP_POINTS
10
{x}
20
{y}
30
{z}
'''
        
        # Add circle (approximated with lines - DXF CIRCLE entity)
        dxf_content += f'''0
CIRCLE
8
GCP_MARKERS
10
{x}
20
{y}
30
{z}
40
2.0
'''
        
        # Add label as TEXT
        gcp_id = gcp.get('id', 'GCP')
        dxf_content += f'''0
TEXT
8
GCP_LABELS
10
{x + 3}
20
{y + 1}
30
{z}
40
1.5
1
{gcp_id}: {z:.2f}m
'''
    
    dxf_content += '''0
ENDSEC
0
EOF
'''
    
    with open(output_path, 'w') as f:
        f.write(dxf_content)
    
    return output_path


def export_to_gpx(gcps: List[Dict], output_path: str,
                  name: str = "GCP Waypoints") -> str:
    """
    Export GCPs to GPX format for GPS devices.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output GPX file
        name: Name for the GPX file
        
    Returns:
        Path to created file
    """
    gpx = f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Topo Survey GCP Tool"
     xmlns="http://www.topografix.com/GPX/1/1">
    <metadata>
        <name>{name}</name>
        <desc>Ground Control Points for Survey</desc>
    </metadata>
'''
    
    for gcp in gcps:
        gcp_id = gcp.get('id', 'GCP')
        x = gcp['x']  # longitude
        y = gcp['y']  # latitude
        z = gcp['elevation']
        
        gpx += f'''
    <wpt lat="{y}" lon="{x}">
        <ele>{z}</ele>
        <name>{gcp_id}</name>
        <desc>Elevation: {z:.3f}m, Type: {gcp.get('type', 'unknown')}</desc>
        <sym>Flag, Red</sym>
    </wpt>
'''
    
    gpx += '''
</gpx>'''
    
    with open(output_path, 'w') as f:
        f.write(gpx)
    
    return output_path


def export_to_pdf_report(gcps: List[Dict], output_path: str,
                         project_name: str = "Topographic Survey",
                         dem_stats: Optional[Dict] = None) -> str:
    """
    Export GCP list to a PDF report.
    
    Args:
        gcps: List of GCP dictionaries
        output_path: Path for output PDF file
        project_name: Name of the project
        dem_stats: Optional DEM statistics to include
        
    Returns:
        Path to created file
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        elements.append(Paragraph(f"<b>{project_name}</b>", styles['Heading1']))
        elements.append(Paragraph("Ground Control Points Report", styles['Heading2']))
        elements.append(Spacer(1, 20))
        
        # Summary
        elements.append(Paragraph(f"Total GCPs: {len(gcps)}", styles['Normal']))
        elements.append(Spacer(1, 10))
        
        # GCP Table
        table_data = [['ID', 'X', 'Y', 'Elevation (m)', 'Slope (°)', 'Type']]
        
        for gcp in gcps:
            table_data.append([
                gcp.get('id', ''),
                f"{gcp['x']:.4f}",
                f"{gcp['y']:.4f}",
                f"{gcp['elevation']:.3f}",
                f"{gcp.get('slope', 0):.1f}",
                gcp.get('type', '')
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        doc.build(elements)
        return output_path
        
    except ImportError:
        # Fallback: Create a simple text report
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w') as f:
            f.write(f"{project_name}\n")
            f.write("Ground Control Points Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total GCPs: {len(gcps)}\n\n")
            f.write(f"{'ID':<10} {'X':>14} {'Y':>14} {'Elev':>10} {'Slope':>8} {'Type':<15}\n")
            f.write("-" * 75 + "\n")
            
            for gcp in gcps:
                f.write(f"{gcp.get('id', ''):<10} "
                       f"{gcp['x']:>14.4f} "
                       f"{gcp['y']:>14.4f} "
                       f"{gcp['elevation']:>10.3f} "
                       f"{gcp.get('slope', 0):>8.1f} "
                       f"{gcp.get('type', ''):<15}\n")
        
        return txt_path
