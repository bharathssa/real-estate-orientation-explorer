# Real Estate Orientation Explorer

A Streamlit dashboard for Australian property investors, analysts, and GIS users. This app helps you fetch and analyse property listings, compute property orientation using GIS data, and create investor-friendly metrics and downloadable reports.

---

## Features

- Fetch property listings for any Australian suburb using the Microburbs API.
- Compute and visualise key investor metrics such as price per square metre, investor readiness score, and amenity features like pool, solar, or outdoor entertainment areas.
- GIS Orientation Module: Upload your own G-NAF, cadastre, and roads datasets to automatically calculate each property's facing direction (for example, north-facing).
- Filter, visualise, and download results as CSV files.
- Interactive data exploration, including filterable tables, orientation breakdowns, and summary charts.
- Works fully offline for GIS orientation with your own spatial files.

---

## Quickstart

1. **Clone this repository** or copy the project folder to your computer.

2. **Install required packages** using pip:


3. **Run the Streamlit app:**


*(Replace `mini_proj.py` with your main Python file if it is named differently.)*

---

## Inputs and File Structure

- Online API: Property data is fetched using suburb and API token (default is "test").
- GIS module files (required for orientation explorer):
 - `cadastre.gpkg` : Parcel/cadastre boundaries (GeoPackage)
 - `gnaf_prop.parquet` : Geocoded address points (Parquet file, must include longitude and latitude columns)
 - `roads.gpkg` : Road centerlines (GeoPackage)

- Typical project structure:


---

## How to Use

- Enter a suburb name (such as "Belmont North") and, if required, an API token.
- Use the sidebar to set filters (minimum bedrooms, outlier hiding, etc.).
- Enable the Orientation Module if you have local GIS files. Enter or browse to the filenames for the required cadastre, address, and roads data.
- The app computes property orientation and displays results in a table and as an orientation summary.
- You can filter results by orientation and download the filtered results as CSV.
- All outputs and data visualisations are available in the Streamlit interface.

---

## Data Sources

- Microburbs API (for online property listing data): https://www.microburbs.com.au/
- G-NAF address data: https://data.gov.au/dataset/geocoded-national-address-file-g-naf
- Cadastre and road data: Typically from local government, state spatial data services, or OpenStreetMap exports.

---

## Customisation Ideas

- Add new investor metrics or visualisations (e.g., price per bedroom, sales trends).
- Join with sales transaction data for premium market analysis.
- Batch export all properties with certain features (e.g., all north-facing homes above 600 sqm).
- Integrate more advanced map features, spatial joins, or clustering.
- Support for additional GIS file formats as needed.

---

## License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

## Author

Bharath Chandran Madhaiyan  
GitHub: https://github.com/bharathssa

---

For any issues, feedback, or questions, please open an issue in this repository.
