# Import libraries
import openeo

# Data Handling
import pandas as pd
import numpy as np
import geopandas

# Spatial Analysis
import rasterio
import shapely
import xarray

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Machine Learning
import sklearn
import xgboost

# Statistical Analysis
import scipy
import statsmodels

# Load GeoTIFF
Sentinel2_Vondel = "Datasets/Sentinel2_Vondel.tiff"
Sentinel2_Vondel_NoTAggregate = "Datasets/Sentinel2_Vondel_NoTAggregate.tif"
Sentinel5_Vondel = "Datasets/Sentinel5_CO_Vondel.tiff"
EVI = "Datasets/result (1) - EVI.tiff"
EVI_GeoJSON = "Datasets/result (2) - EVI - geoJSON.tiff"
NDVI = "Datasets/result (3) - NDVI.tiff"
MSI = "Datasets/result (6) - MSI.tiff"
SAVI = "Datasets/result (7) - SAVI.tiff"
Sentinel5And2_Amsterdam = "Datasets/Sentinel5And2_Amsterdam.tif"
Sentinel2_Ams = "Datasets/Sentinel2_Ams.tif"
Sentinel5_Ams = "Datasets/Sentinel5_Ams.tif"

def inspect_metadata(filename):
    try:
        with rasterio.open(filename) as src:
            metadata = {
                "width": src.width,
                "height": src.height,
                "count": src.count,  # Number of bands
                "crs": src.crs,  # Coordinate Reference System
                "transform": src.transform,  # Affine transformation
                "bounds": src.bounds,  # Spatial extent
                "dtype": src.dtypes,  # Data type of the raster
            }
    except Exception as e:
        metadata = {"error": str(e)}

    return metadata

def plot_bands(filename):
    with rasterio.open(filename) as src:
        for i in range(1, src.count + 1):

            band = src.read(i)

            plt.figure(figsize=(10, 5))
            plt.title("Band Visualization")
            plt.imshow(band, cmap="viridis")
            plt.colorbar(label="Pixel Values")
            plt.xlabel("X Pixels")
            plt.ylabel("Y Pixels")
            plt.show()

def band_statistics(filename):
    raster_stats = {}

    with rasterio.open(filename) as src:
        for i in range(1, src.count + 1):  # Loop through bands
            band_data = src.read(i, masked=True)  # Read band data as a masked array to handle NoData values

            # Calculate statistics
            stats = {
                "min": float(np.min(band_data)),          # Minimum value
                "max": float(np.max(band_data)),          # Maximum value
                "mean": float(np.mean(band_data)),        # Mean
                "median": float(np.median(band_data)),    # Median
                "std": float(np.std(band_data)),          # Standard deviation
                "nodata": np.sum(band_data.mask)          # Count of NoData values
            }

            raster_stats[f"Band {i}"] = stats

    return raster_stats

plot_bands(NDVI)
band_statistics(Sentinel5_Ams)




class GeoTIFFToCSV:
    def __init__(self, file_path):
        """
        Initialize the GeoTIFFToCSV class with the file path.
        """
        self.file_path = file_path
        self.data = None
        self.metadata = None

    def load_geotiff(self):
        """
        Load the GeoTIFF file and extract data and metadata.
        """
        with rasterio.open(self.file_path) as src:
            self.data = src.read()  # Read all bands
            self.metadata = src.meta  # Metadata such as spatial resolution and CRS
            self.dates = src.descriptions  # Band descriptions (dates if available)
        print("GeoTIFF loaded successfully.")

    def aggregate_to_timeseries(self):
        """
        Aggregate the GeoTIFF data into a time series (mean values for each band).
        """
        if self.data is None:
            raise ValueError("GeoTIFF data is not loaded. Call load_geotiff() first.")

        # Calculate mean NDVI for each time step (band)
        time_series = []
        for i, band in enumerate(self.data):
            mean_value = np.nanmean(band)  # Mean NDVI for the band (ignoring NaN values)
            time_series.append({
                "Date": self.dates[i] if self.dates else f"Band {i+1}",
                "Mean NDVI": mean_value
            })

        # Convert to DataFrame
        self.time_series_df = pd.DataFrame(time_series)
        print("Time series aggregated successfully.")

    def save_to_csv(self, output_path):
        """
        Save the time series data to a CSV file.
        """
        if self.time_series_df is None:
            raise ValueError("Time series data is not aggregated. Call aggregate_to_timeseries() first.")

        self.time_series_df.to_csv(output_path, index=False)
        print(f"Time series saved to {output_path}.")


Sentinel2_Vondel_Csv = GeoTIFFToCSV(Sentinel2_Vondel_NoTAggregate)

Sentinel2_Vondel_Csv.load_geotiff()

Sentinel2_Vondel_Csv.aggregate_to_timeseries()

Sentinel2_Vondel_Csv.save_to_csv("Sentinel2_Vondel_NoTAggregate_Csv.csv")