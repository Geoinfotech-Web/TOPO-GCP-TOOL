# 🗺️ Topographic Survey GCP Generation Tool

A unified Python-based tool that automates the entire topographic analysis workflow for drone surveys. Upload a DEM file, and get slope maps, contours, hydrological analysis, and ground control point markers in one click.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

##  Features

- **One-Click Processing**: Upload DEM, get all outputs in 5-10 minutes
- **Zero Licensing Costs**: Built on free, open-source tools (GDAL, Python)
- **Repeatable & Standardized**: Same workflow for every survey
- **Offline-Capable**: Runs on any computer, no cloud dependency
- **Multiple Export Formats**: CSV, KML, Shapefile, DXF

## Output Datasets

| Dataset | Description | Format |
|---------|-------------|--------|
| **Slope** | Terrain steepness in degrees | GeoTIFF |
| **Contours** | Elevation lines at regular intervals | GeoJSON |
| **Flow Direction** | D8 water flow patterns | GeoTIFF |
| **Flow Accumulation** | Upstream contributing area | GeoTIFF |
| **Watersheds** | Catchment basin boundaries | GeoTIFF |
| **Stream Network** | Extracted drainage lines | GeoJSON |
| **GCP Markers** | Ground control point coordinates | CSV, KML, SHP, DXF |

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Recommended)

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub
4. Select this repository and `app.py` as the main file
5. Click "Deploy!" - Your app will be live in minutes!

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/topo-gcp-tool.git
cd topo-gcp-tool

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment Options

### Streamlit Cloud (Free - Best Option)
- **Pros**: Free, easy setup, handles GDAL dependencies automatically
- **Setup Time**: ~5 minutes
- **URL**: `https://your-app-name.streamlit.app`

### Hugging Face Spaces (Free)
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Streamlit" as the SDK
3. Upload all files or connect to GitHub
4. Add a `packages.txt` file with: `gdal-bin libgdal-dev`

### Railway (Free Tier)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Docker Deployment
```dockerfile
# Dockerfile included in the repository
docker build -t topo-gcp-tool .
docker run -p 8501:8501 topo-gcp-tool
```

## 📁 Project Structure

```
topo-gcp-tool/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt           # System packages (for Streamlit Cloud)
├── README.md              # This file
├── Dockerfile             # Docker configuration
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── modules/
    ├── __init__.py
    ├── dem_processor.py   # DEM loading and saving
    ├── slope_analysis.py  # Slope, aspect, curvature
    ├── contour_generator.py # Contour line generation
    ├── hydrology.py       # Flow direction, accumulation, watersheds
    ├── gcp_generator.py   # GCP marker generation
    └── exporters.py       # CSV, KML, SHP, DXF export
```

## ⚙️ Configuration Options

### Contour Settings
- **Contour Interval**: 0.5m - 20m (default: 2m)

### GCP Settings
- **GCP Spacing**: 50m - 500m (default: 100m)
- **Placement Strategy**:
  - Grid Pattern: Regular grid distribution
  - Terrain-Adaptive: Places points at terrain features
  - Edge + Interior: Prioritizes edge coverage

### Hydrology Settings
- **Stream Threshold**: 50 - 1000 (default: 200)

### Export Formats
- CSV (Coordinates)
- KML (Google Earth)
- Shapefile (GIS)
- DXF (CAD)

## 🔧 System Requirements

### Minimum
- Python 3.9+
- 4GB RAM
- 2GB free disk space

### Recommended
- Python 3.10+
- 8GB RAM
- 10GB free disk space
- Multi-core processor

## 📊 Supported Input Formats

- GeoTIFF (.tif, .tiff)
- Must be a single-band elevation raster
- Projected coordinate system recommended (UTM)
- Accurate georeferencing required

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Geospatial processing powered by [GDAL](https://gdal.org/), [Rasterio](https://rasterio.readthedocs.io/), and [Fiona](https://fiona.readthedocs.io/)
- Developed for Geoinfotech Kaduna Drone Topographic Survey Project

## 📞 Support

For issues and feature requests, please open an issue on GitHub.

---

**Made with ❤️ for the surveying community**
