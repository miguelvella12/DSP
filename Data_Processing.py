import glob
import os
import rasterio
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def rename_files(directory, old_prefix, new_prefix):
    """
    Rename files in a directory by replacing prefix

    :param directory: Path to the directory
    :param old_prefix: Prefix to be replaced
    :param new_prefix: Prefix to replace
    """

    try:
        files = os.listdir(directory)

        for file in files:
            # Check if file starts with old prefix
            if file.startswith(old_prefix):
                # Construct new file name
                new_name = file.replace(old_prefix, new_prefix, 1) # Replace only first occurrence

                # Rename file
                old_path = os.path.join(directory, file)
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed {file} to {new_name}")

        print("All files renamed successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

    return

def convert_geotiff_to_csv(directory, output_csv):
    """
    Combining GeoTIFF files to convert into CSV file with headers Date, Coordinates and Value

    :param directory: Path to the directory containing GeoTIFF files
    :param output_csv: Path to the output CSV file
    """

    data_rows = []

    # Loop through all files in directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            # Parse date from filename (assumes format XXX_XXX_YYY-MM-DDZ.tif)
            date = filename.split("_")[2].replace("Z.tif","") # Remove Z.tif at end of file

            # Open GeoTIFF
            file_path = os.path.join(directory, filename)
            with rasterio.open(file_path) as src:
                # Read first band
                band = src.read(1)

                # Get coordinates and pixel values
                for row_idx in range(band.shape[0]):
                    for col_idx in range(band.shape[1]):
                        value = band[row_idx, col_idx]

                        # Skip NoData values
                        if value is None or value == src.nodata:
                            continue

                        # Get coordinates
                        x, y = src.xy(row_idx, col_idx)

                        # Append to data rows
                        data_rows.append({
                            "Date": date,
                            "Coordinates": f"{x}, {y}",
                            "Value": value
                        })

    df = pd.DataFrame(data_rows)

    df.to_csv(output_csv, index=False)
    print(f"CSV file written to {output_csv}")

    return

def change_sentinel5_csv_structure(s5_file_path, s2_file_path, output_csv_path):
    """
    Change the structure of Sentinel-5P csv to match Sentinel-2 (due to uniform CO values per year)

    :param s5_file_path: Path to the Sentinel-5P csv file
    :param s2_file_path: Path to the Sentinel-2 csv file
    :param output_csv_path: Path to the output CSV file
    :return:
    """

    s5_data = pd.read_csv(s5_file_path)
    s2_data = pd.read_csv(s2_file_path)

    # Create a mapping for Sentinel-5P values for each date
    s5_date_mapping = s5_data.set_index("Date")['Value'].to_dict()

    # Replicate Sentinel-5P data to match Sentinel-2 structure
    new_s5_date_mapping = [
        {
            'Date': s2_row['Date'],
            'Coordinates': s2_row['Coordinates'],
            'Value': s5_date_mapping.get(s2_row['Date'], None)
        }
        for _, s2_row in s2_data.iterrows()
    ]

    # Create new DataFrame for new Sentinel-5P data
    new_s5_df = pd.DataFrame(new_s5_date_mapping)

    # Save
    new_s5_df.to_csv(output_csv_path, index=False)
    print(f"CSV file written to {output_csv_path}")

    return

def merge_aqi_files(input_directory, output_directory, green_space):
    """
    Process Sentinel-5P monthly pollutant GeoTIFFS, calculating AQI by taking
    maximum pollutant value at each coordinate

    :param input_directory: Path to folder containing pollutant GeoTIFF subdirectories
    :param output_directory: Path to output directory
    :param green_space: Name of green space
    :return:
    """

    # Define pollutant folders
    pollutants = [green_space + "_" + "CO", green_space + "_" +  "NO2", green_space + "_" +  "O3",
                  green_space + "_" + "SO2", green_space + "_" +  "AER"]

    # Get list of monthly files
    monthly_files = {p: sorted(glob.glob(os.path.join(input_directory, p, "*.tif"))) for p in pollutants}

    placeholder_value = 0  # Assumes missing data means no pollution

    # Process AQI per month
    for month_idx in range(12):
        pollutant_layers = []
        try:
            for pollutant in pollutants:
                file_path = monthly_files[pollutant][month_idx]
                with rasterio.open(file_path) as src:
                    band = src.read(1) # Read AQI values per pollutant
                    pollutant_layers.append(band)
                    meta = src.meta # Save metadata
        except IndexError:
            print(f"Warning: Missing file for {pollutant} in month {month_idx+1}")
            if pollutant_layers:
                data_shape = pollutant_layers[0].shape  # Use existing pollutant data to infer shape
                missing_data = np.full(data_shape, placeholder_value, dtype=np.float32)
            else:
                raise ValueError(f"No valid data found for pollutant {pollutant}")
            pollutant_layers.append(missing_data)

        # Take max value of pollutants to calculate AQI
        aqi = np.maximum.reduce(pollutant_layers)

        # Update metadata
        meta.update(dtype=rasterio.float32)

        # Save final AQI raster
        if month_idx+1 <= 9:
            output_file_path = os.path.join(output_directory, f"{green_space}_AirQualityIndex_2024-0{month_idx+1}-01Z.tif")
        else:
            output_file_path = os.path.join(output_directory, f"{green_space}_AirQualityIndex_2024-{month_idx+1}-01Z.tif")
        with rasterio.open(output_file_path, "w", **meta) as dst:
            dst.write(aqi, 1)
            print(f"Saved AQI raster for month {month_idx+1} to {output_file_path}")

    return

def spatial_match(ndvi_file, file_to_match, output_file):
    """
    Spatially matches csv file with NDVI csv based on spatial proximity. The resulting csv contains the value
    of the nearest NDVI value for each coordinate in the file_to_match csv.

    :param ndvi_file: Path to NDVI csv file
    :param file_to_match: Path to csv file to match, i.e. Soil Moisture and Land Temperature
    :param output_file: Path to output file
    :return:
    """

    # Load datasets
    file_to_match_df = pd.read_csv(file_to_match)
    ndvi_df = pd.read_csv(ndvi_file)

    # Convert 'Date' columns to datetime
    file_to_match_df['Date'] = pd.to_datetime(file_to_match_df['Date'])
    ndvi_df['Date'] = pd.to_datetime(ndvi_df['Date'])

    # Convert coordinates to numeric values for spatial matching
    file_to_match_df[['X', 'Y']] = file_to_match_df['Coordinates'].str.split(',', expand=True).astype(float)
    ndvi_df[['X', 'Y']] = ndvi_df['Coordinates'].str.split(',', expand=True).astype(float)

    # Build a KDTree for NDVI coordinates
    tree = cKDTree(ndvi_df[['X', 'Y']])

    # Find the nearest NDVI value for each soil moisture location
    distances, indices = tree.query(file_to_match_df[['X', 'Y']])

    # Assign matched NDVI values and distance
    file_to_match_df['Nearest_NDVI'] = ndvi_df.iloc[indices]['Value'].values

    # Drop temporary coordinate columns
    file_to_match_df.drop(columns=['X', 'Y'], inplace=True)

    # Save the matched dataset
    file_to_match_df.to_csv(output_file, index=False)

    print(f"Saved matched dataset to {output_file}")

    return

old_prefix = "openEO"

# ########################################################
# # Data Processing for NDVI (Sentinel-2) Files
# ########################################################
#
#
## VONDEL
#
# Update the `directory` path to the folder containing your files
# directory = "Datasets/Sentinel-2/Vondel_NDVI"
# # new_prefix = "Vondel_NDVI"
# # rename_files(directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv"
# convert_geotiff_to_csv(directory, output_csv)
#
# ## AMSTEL
#
# # Update the `directory` path to the folder containing your files
# directory = "Datasets/Sentinel-2/Amstel_NDVI"
# new_prefix = "Amstel_NDVI"
# rename_files(directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv"
# convert_geotiff_to_csv(directory, output_csv)
#
# ## WESTER
#
# # Update the `directory` path to the folder containing your files
# directory = "Datasets/Sentinel-2/Wester_NDVI"
# new_prefix = "Wester_NDVI"
# rename_files(directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv"
# convert_geotiff_to_csv(directory, output_csv)
#
# ## REMBRANDT
#
# # Update the `directory` path to the folder containing your files
# directory = "Datasets/Sentinel-2/Rembrandt_NDVI"
# new_prefix = "Rembrandt_NDVI"
# rename_files(directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv"
# convert_geotiff_to_csv(directory, output_csv)
#
#
# ########################################################
# # Data Processing for AQI (Sentinel-5P) Files
# ########################################################
#
# ## VONDEL
#
# vondel_aer_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_AER"
# vondel_co_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_CO"
# vondel_no2_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_NO2"
# vondel_o3_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_O3"
# vondel_so2_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_SO2"
#
# # Rename files
# new_prefix = "Vondel_AER"
# rename_files(vondel_aer_directory, old_prefix, new_prefix)
# new_prefix = "Vondel_CO"
# rename_files(vondel_co_directory, old_prefix, new_prefix)
# new_prefix = "Vondel_NO2"
# rename_files(vondel_no2_directory, old_prefix, new_prefix)
# new_prefix = "Vondel_O3"
# rename_files(vondel_o3_directory, old_prefix, new_prefix)
# new_prefix = "Vondel_SO2"
# rename_files(vondel_o3_directory, old_prefix, new_prefix)
#
# # Calculate AQI
# input_directory = "Datasets/Sentinel-5P/Vondelpark"
# output_directory = "Datasets/Sentinel-5P/Vondelpark/Vondel_AirQualityIndex"
# merge_aqi_files(input_directory, output_directory, "Vondel")
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-5P/Vondelpark/Vondel_AirQualityIndex/Vondel_AirQualityIndex_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# #Change structure of file to match sentinel-2 file
# change_sentinel5_csv_structure(s5_file_path="Datasets/Sentinel-5P/Vondelpark/Vondel_AirQualityIndex/Vondel_AirQualityIndex_csv.csv",
#  s2_file_path="Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv",
#  output_csv_path="Datasets/Sentinel-5P/Vondelpark/Vondel_AirQualityIndex/Vondel_AirQualityIndex_csv.csv")
#
# ## AMSTEL
# amstel_aer_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_AER"
# amstel_co_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_CO"
# amstel_no2_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_NO2"
# amstel_o3_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_O3"
# amstel_so2_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_SO2"
#
# # Rename files
# new_prefix = "Amstel_AER"
# rename_files(amstel_aer_directory, old_prefix, new_prefix)
# new_prefix = "Amstel_CO"
# rename_files(amstel_co_directory, old_prefix, new_prefix)
# new_prefix = "Amstel_NO2"
# rename_files(amstel_no2_directory, old_prefix, new_prefix)
# new_prefix = "Amstel_O3"
# rename_files(amstel_o3_directory, old_prefix, new_prefix)
# new_prefix = "Amstel_SO2"
# rename_files(amstel_so2_directory, old_prefix, new_prefix)
#
# # Calculate AQI
# input_directory = "Datasets/Sentinel-5P/Amstelpark"
# output_directory = "Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex"
# merge_aqi_files(input_directory, output_directory, "Amstel")
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex/Amstel_AirQualityIndex_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# #Change structure of file to match sentinel-2 file
# change_sentinel5_csv_structure(s5_file_path="Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex/Amstel_AirQualityIndex_csv.csv",
#  s2_file_path="Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
#  output_csv_path="Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex/Amstel_AirQualityIndex_csv.csv")
#
# ## REMBRANDT
# rembrandt_aer_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AER"
# rembrandt_co_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_CO"
# rembrandt_no2_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_NO2"
# rembrandt_o3_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_O3"
# rembrandt_so2_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_SO2"
#
# # Rename files
# new_prefix = "Rembrandt_AER"
# rename_files(rembrandt_aer_directory, old_prefix, new_prefix)
# new_prefix = "Rembrandt_CO"
# rename_files(rembrandt_co_directory, old_prefix, new_prefix)
# new_prefix = "Rembrandt_NO2"
# rename_files(rembrandt_no2_directory, old_prefix, new_prefix)
# new_prefix = "Rembrandt_O3"
# rename_files(rembrandt_o3_directory, old_prefix, new_prefix)
# new_prefix = "Rembrandt_SO2"
# rename_files(rembrandt_so2_directory, old_prefix, new_prefix)
#
# # Calculate AQI
# input_directory = "Datasets/Sentinel-5P/Rembrandtpark"
# output_directory = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex"
# merge_aqi_files(input_directory, output_directory, "Rembrandt")
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# #Change structure of file to match sentinel-2 file
# change_sentinel5_csv_structure(s5_file_path="Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv",
#  s2_file_path="Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
#  output_csv_path="Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv")
#
# ## WESTER
# wester_aer_directory = "Datasets/Sentinel-5P/Westerpark/Wester_AER"
# wester_co_directory = "Datasets/Sentinel-5P/Westerpark/Wester_CO"
# wester_no2_directory = "Datasets/Sentinel-5P/Westerpark/Wester_NO2"
# wester_o3_directory = "Datasets/Sentinel-5P/Westerpark/Wester_O3"
# wester_so2_directory = "Datasets/Sentinel-5P/Westerpark/Wester_SO2"
#
# # Rename files
# new_prefix = "Wester_AER"
# rename_files(wester_aer_directory, old_prefix, new_prefix)
# new_prefix = "Wester_CO"
# rename_files(wester_co_directory, old_prefix, new_prefix)
# new_prefix = "Wester_NO2"
# rename_files(wester_no2_directory, old_prefix, new_prefix)
# new_prefix = "Wester_O3"
# rename_files(wester_o3_directory, old_prefix, new_prefix)
# new_prefix = "Wester_SO2"
# rename_files(wester_so2_directory, old_prefix, new_prefix)
#
# # Calculate AQI
# input_directory = "Datasets/Sentinel-5P/Westerpark"
# output_directory = "Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex"
# merge_aqi_files(input_directory, output_directory, "Wester")
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex/Wester_AirQualityIndex_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# #Change structure of file to match sentinel-2 file
# change_sentinel5_csv_structure(s5_file_path="Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex/Wester_AirQualityIndex_csv.csv",
#                                s2_file_path="Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
#                                output_csv_path="Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex/Wester_AirQualityIndex_csv.csv")
#
# ########################################################
# # Data Processing for Soil Moisture (Sentinel-1) Files
# ########################################################
#
## VONDEL
#
# Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Vondelpark/Vondel_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Vondelpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Assign nearest NDVI value for each coordinate
# spatial_match("Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv",
#               "Datasets/Sentinel-1/Vondelpark/Vondel_SoilMoisture_csv.csv",
#               "Datasets/Sentinel-1/Vondelpark/Vondel_SoilMoisture_csv.csv")
#
# ## AMSTEL
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Amstelpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Assign nearest NDVI value for each coordinate
# spatial_match("Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
#               "Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv",
#               "Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv")
#
# ## REMBRANDT
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Rembrandtpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Assign nearest NDVI value for each coordinate
# spatial_match("Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
#               "Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv",
#               "Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv")
#
# ## WESTER
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Westerpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Assign nearest NDVI value for each coordinate
# spatial_match("Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
#               "Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv",
#               "Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv")
#
# #########################################################
# # Data Processing for Land Surface Temp (Landsat-8) Files
# #########################################################
#
## VONDEL

# Convert GeoTIFF files to .csv file
output_csv = "Datasets/Landsat-8/Vondelpark/Vondel_LandTemp_csv.csv"
output_directory = "Datasets/Landsat-8/Vondelpark"
convert_geotiff_to_csv(output_directory, output_csv)

# Assign nearest NDVI value for each coordinate
spatial_match("Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv",
              "Datasets/Landsat-8/Vondelpark/Vondel_LandTemp_csv.csv",
              "Datasets/Landsat-8/Vondelpark/Vondel_LandTemp_csv.csv")

## AMSTEL

# Convert GeoTIFF files to .csv file
output_csv = "Datasets/Landsat-8/Amstelpark/Amstel_LandTemp_csv.csv"
output_directory = "Datasets/Landsat-8/Amstelpark"
convert_geotiff_to_csv(output_directory, output_csv)

# Assign nearest NDVI value for each coordinate
spatial_match("Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
              "Datasets/Landsat-8/Amstelpark/Amstel_LandTemp_csv.csv",
              "Datasets/Landsat-8/Amstelpark/Amstel_LandTemp_csv.csv")

## WESTER

# Convert GeoTIFF files to .csv file
output_csv = "Datasets/Landsat-8/Westerpark/Wester_LandTemp_csv.csv"
output_directory = "Datasets/Landsat-8/Westerpark"
convert_geotiff_to_csv(output_directory, output_csv)

# Assign nearest NDVI value for each coordinate
spatial_match("Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
              "Datasets/Landsat-8/Westerpark/Wester_LandTemp_csv.csv",
              "Datasets/Landsat-8/Westerpark/Wester_LandTemp_csv.csv")

## REMBRANDT

# Convert GeoTIFF files to .csv file
output_csv = "Datasets/Landsat-8/Rembrandtpark/Rembrandt_LandTemp_csv.csv"
output_directory = "Datasets/Landsat-8/Rembrandtpark"
convert_geotiff_to_csv(output_directory, output_csv)

# Assign nearest NDVI value for each coordinate
spatial_match("Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
              "Datasets/Landsat-8/Rembrandtpark/Rembrandt_LandTemp_csv.csv",
              "Datasets/Landsat-8/Rembrandtpark/Rembrandt_LandTemp_csv.csv")