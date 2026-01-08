"""
Topographic Survey GCP Generation Tool
=======================================
A unified Python-based tool that automates the entire topographic analysis workflow.

Project: Geoinfotech - Kaduna Drone Topographic Survey
Author: Geoinfotech Team
"""

import streamlit as st
import numpy as np
import tempfile
import os
import zipfile
from io import BytesIO
from pathlib import Path

# Import processing modules
from modules.dem_processor import DEMProcessor
from modules.slope_analysis import calculate_slope
from modules.contour_generator import generate_contours
from modules.hydrology import calculate_flow_direction, calculate_flow_accumulation, delineate_watersheds, extract_streams
from modules.gcp_generator import generate_gcp_markers
from modules.exporters import export_to_csv, export_to_kml, export_to_shapefile, export_to_dxf

# Page configuration
st.set_page_config(
    page_title="Topo Survey GCP Tool",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary: #0F4C3A;
        --secondary: #1A6B50;
        --accent: #2DD4A3;
        --dark: #0A1F1A;
        --light: #E8F5F0;
        --warning: #F59E0B;
        --danger: #EF4444;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0A1F1A 0%, #0F2922 50%, #0A1F1A 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(45, 212, 163, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    .stat-card {
        background: linear-gradient(145deg, rgba(15, 76, 58, 0.6), rgba(26, 107, 80, 0.4));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(45, 212, 163, 0.15);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stat-card h3 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        color: var(--accent);
        margin: 0;
    }
    
    .stat-card p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        margin: 0.5rem 0 0 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .process-step {
        background: rgba(15, 76, 58, 0.3);
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid var(--accent);
        margin: 0.75rem 0;
    }
    
    .process-step h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #FFFFFF;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .process-step p {
        font-family: 'IBM Plex Sans', sans-serif;
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .success-box {
        background: linear-gradient(145deg, rgba(45, 212, 163, 0.15), rgba(45, 212, 163, 0.05));
        border: 1px solid rgba(45, 212, 163, 0.4);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(145deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F2922 0%, #0A1F1A 100%);
        border-right: 1px solid rgba(45, 212, 163, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--accent);
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid rgba(45, 212, 163, 0.2);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent), #1A9B6C);
        color: #0A1F1A;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(45, 212, 163, 0.4);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 76, 58, 0.2);
        border: 2px dashed rgba(45, 212, 163, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent), #1A9B6C);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        background: rgba(15, 76, 58, 0.3);
        border-radius: 8px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 76, 58, 0.2);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        background: transparent;
        border-radius: 6px;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary);
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Topographic Survey GCP Generation Tool</h1>
        <p>Automated terrain analysis and ground control point generation for drone surveys</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### 📏 Contour Settings")
        contour_interval = st.slider(
            "Contour Interval (m)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5,
            help="Vertical distance between contour lines"
        )
        
        st.markdown("### 📍 GCP Settings")
        gcp_spacing = st.slider(
            "GCP Spacing (m)",
            min_value=50,
            max_value=500,
            value=100,
            step=25,
            help="Distance between generated GCP markers"
        )
        
        gcp_strategy = st.selectbox(
            "GCP Placement Strategy",
            ["Grid Pattern", "Terrain-Adaptive", "Edge + Interior"],
            help="Method for distributing GCP markers across the survey area"
        )
        
        st.markdown("### 🌊 Hydrology Settings")
        flow_threshold = st.slider(
            "Stream Threshold",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Minimum flow accumulation to define a stream"
        )
        
        st.markdown("### 📁 Export Formats")
        export_csv = st.checkbox("CSV (Coordinates)", value=True)
        export_kml = st.checkbox("KML (Google Earth)", value=True)
        export_shp = st.checkbox("Shapefile (GIS)", value=True)
        export_dxf = st.checkbox("DXF (CAD)", value=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; opacity: 0.7; font-size: 0.8rem;">
            <p>Geoinfotech</p>
            <p>Kaduna Drone Survey Project</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>7</h3>
            <p>Output Datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>4</h3>
            <p>Export Formats</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>100%</h3>
            <p>Open Source</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3>~5min</h3>
            <p>Processing Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File upload section
    st.markdown("### 📤 Upload DEM File")
    
    uploaded_file = st.file_uploader(
        "Upload your Digital Elevation Model (GeoTIFF format)",
        type=['tif', 'tiff', 'geotiff'],
        help="Supported formats: GeoTIFF (.tif, .tiff)"
    )
    
    if uploaded_file is not None:
        # Create a temporary file to save the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.markdown("""
        <div class="success-box">
            <strong>✅ File uploaded successfully!</strong><br>
            Ready to process your DEM data.
        </div>
        """, unsafe_allow_html=True)
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("File Size", f"{file_size:.2f} MB")
        
        # Process button
        if st.button("🚀 Start Processing", use_container_width=True):
            
            # Create output directory
            output_dir = tempfile.mkdtemp()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load DEM
                status_text.text("📂 Loading DEM file...")
                progress_bar.progress(5)
                
                processor = DEMProcessor(tmp_path)
                dem_data, transform, crs, bounds = processor.load_dem()
                
                st.markdown("""
                <div class="process-step">
                    <h4>✅ DEM Loaded Successfully</h4>
                    <p>Raster data extracted and georeferencing information captured.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display DEM statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min Elevation", f"{np.nanmin(dem_data):.2f} m")
                with col2:
                    st.metric("Max Elevation", f"{np.nanmax(dem_data):.2f} m")
                with col3:
                    st.metric("Resolution", f"{abs(transform[0]):.2f} m")
                with col4:
                    st.metric("Size", f"{dem_data.shape[1]} x {dem_data.shape[0]}")
                
                progress_bar.progress(15)
                
                # Step 2: Calculate Slope
                status_text.text("📐 Calculating slope...")
                slope_data = calculate_slope(dem_data, transform)
                slope_path = os.path.join(output_dir, "slope.tif")
                processor.save_raster(slope_data, slope_path, "slope")
                
                st.markdown("""
                <div class="process-step">
                    <h4>✅ Slope Analysis Complete</h4>
                    <p>Terrain steepness calculated for planning and analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(25)
                
                # Step 3: Generate Contours
                status_text.text("🗺️ Generating contours...")
                contours_path = os.path.join(output_dir, "contours.geojson")
                contour_count = generate_contours(dem_data, transform, crs, contour_interval, contours_path)
                
                st.markdown(f"""
                <div class="process-step">
                    <h4>✅ Contours Generated</h4>
                    <p>{contour_count} contour lines created at {contour_interval}m interval.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(40)
                
                # Step 4: Flow Direction
                status_text.text("🧭 Calculating flow direction...")
                flow_dir = calculate_flow_direction(dem_data)
                flow_dir_path = os.path.join(output_dir, "flow_direction.tif")
                processor.save_raster(flow_dir, flow_dir_path, "flow_direction")
                
                st.markdown("""
                <div class="process-step">
                    <h4>✅ Flow Direction Calculated</h4>
                    <p>Water movement patterns across terrain analyzed.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(55)
                
                # Step 5: Flow Accumulation
                status_text.text("💧 Calculating flow accumulation...")
                flow_acc = calculate_flow_accumulation(flow_dir)
                flow_acc_path = os.path.join(output_dir, "flow_accumulation.tif")
                processor.save_raster(flow_acc, flow_acc_path, "flow_accumulation")
                
                st.markdown("""
                <div class="process-step">
                    <h4>✅ Flow Accumulation Complete</h4>
                    <p>Drainage patterns and stream networks identified.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(65)
                
                # Step 6: Watershed Delineation
                status_text.text("🏔️ Delineating watersheds...")
                watersheds = delineate_watersheds(flow_dir, flow_acc)
                watersheds_path = os.path.join(output_dir, "watersheds.tif")
                processor.save_raster(watersheds, watersheds_path, "watersheds")
                
                st.markdown("""
                <div class="process-step">
                    <h4>✅ Watersheds Delineated</h4>
                    <p>Catchment areas and basin boundaries mapped.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(75)
                
                # Step 7: Stream Extraction
                status_text.text("🌊 Extracting stream network...")
                streams_path = os.path.join(output_dir, "streams.geojson")
                stream_count = extract_streams(flow_acc, flow_dir, transform, crs, flow_threshold, streams_path)
                
                st.markdown(f"""
                <div class="process-step">
                    <h4>✅ Stream Network Extracted</h4>
                    <p>{stream_count} stream segments identified.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(85)
                
                # Step 8: Generate GCP Markers
                status_text.text("📍 Generating GCP markers...")
                gcp_data = generate_gcp_markers(
                    dem_data, 
                    transform, 
                    crs, 
                    spacing=gcp_spacing,
                    strategy=gcp_strategy.lower().replace(" ", "_").replace("+", "_")
                )
                
                # Export GCPs in selected formats
                gcp_files = []
                
                if export_csv:
                    csv_path = os.path.join(output_dir, "gcp_markers.csv")
                    export_to_csv(gcp_data, csv_path)
                    gcp_files.append(csv_path)
                
                if export_kml:
                    kml_path = os.path.join(output_dir, "gcp_markers.kml")
                    export_to_kml(gcp_data, kml_path)
                    gcp_files.append(kml_path)
                
                if export_shp:
                    shp_path = os.path.join(output_dir, "gcp_markers.shp")
                    export_to_shapefile(gcp_data, shp_path, crs)
                    gcp_files.append(shp_path)
                
                if export_dxf:
                    dxf_path = os.path.join(output_dir, "gcp_markers.dxf")
                    export_to_dxf(gcp_data, dxf_path)
                    gcp_files.append(dxf_path)
                
                st.markdown(f"""
                <div class="process-step">
                    <h4>✅ GCP Markers Generated</h4>
                    <p>{len(gcp_data)} ground control points created using {gcp_strategy} strategy.</p>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                # Create ZIP file with all outputs
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zip_file.write(file_path, arcname)
                
                zip_buffer.seek(0)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Success message and download
                st.success("🎉 All processing complete! Download your results below.")
                
                # Results summary
                st.markdown("### 📊 Processing Summary")
                
                results_col1, results_col2 = st.columns(2)
                
                with results_col1:
                    st.markdown("""
                    **Raster Outputs:**
                    - `slope.tif` - Terrain steepness
                    - `flow_direction.tif` - Water flow patterns
                    - `flow_accumulation.tif` - Drainage accumulation
                    - `watersheds.tif` - Basin boundaries
                    """)
                
                with results_col2:
                    st.markdown(f"""
                    **Vector Outputs:**
                    - `contours.geojson` - {contour_count} contour lines
                    - `streams.geojson` - {stream_count} stream segments
                    - `gcp_markers.*` - {len(gcp_data)} GCP points
                    """)
                
                # Download button
                st.download_button(
                    label="📥 Download All Results (ZIP)",
                    data=zip_buffer,
                    file_name="topographic_survey_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                # GCP Preview Table
                st.markdown("### 📍 GCP Preview")
                import pandas as pd
                gcp_df = pd.DataFrame(gcp_data)
                st.dataframe(
                    gcp_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
            
            finally:
                # Cleanup temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("### 📋 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="process-step">
                <h4>1️⃣ Upload DEM</h4>
                <p>Upload your Digital Elevation Model in GeoTIFF format from drone photogrammetry.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="process-step">
                <h4>2️⃣ Configure Parameters</h4>
                <p>Adjust contour intervals, GCP spacing, and export formats in the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="process-step">
                <h4>3️⃣ Process</h4>
                <p>Click "Start Processing" and wait 5-10 minutes for automated analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="process-step">
                <h4>4️⃣ Download Results</h4>
                <p>Get all outputs in a single ZIP file ready for your GIS workflow.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
                <strong>💡 Tip:</strong> For best results, ensure your DEM has accurate georeferencing and a projected coordinate system (UTM recommended).
            </div>
            """, unsafe_allow_html=True)
        
        # Output datasets info
        st.markdown("### 📦 Output Datasets")
        
        tabs = st.tabs(["Terrain Analysis", "Hydrology", "GCP Markers"])
        
        with tabs[0]:
            st.markdown("""
            | Dataset | Description | Format |
            |---------|-------------|--------|
            | **Slope** | Terrain steepness in degrees | GeoTIFF |
            | **Contours** | Elevation lines at regular intervals | GeoJSON |
            """)
        
        with tabs[1]:
            st.markdown("""
            | Dataset | Description | Format |
            |---------|-------------|--------|
            | **Flow Direction** | D8 flow direction grid | GeoTIFF |
            | **Flow Accumulation** | Upstream contributing area | GeoTIFF |
            | **Watersheds** | Catchment basin boundaries | GeoTIFF |
            | **Stream Network** | Extracted drainage lines | GeoJSON |
            """)
        
        with tabs[2]:
            st.markdown("""
            | Format | Use Case |
            |--------|----------|
            | **CSV** | Spreadsheet import, custom processing |
            | **KML** | Google Earth visualization |
            | **Shapefile** | GIS software (QGIS, ArcGIS) |
            | **DXF** | CAD software (AutoCAD, Civil 3D) |
            """)


if __name__ == "__main__":
    main()
