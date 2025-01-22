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

def change_csv_structure(factor_file_path, index_file_path, output_csv_path):
    """
    Change the structure of the environmental factor CSV to match NDVI values according to spatial proximity and date.
    In this way all csv files will have same amount of rows

    :param factor_file_path: Path to the CSV factor file
    :param index_file_path: Path to the CSV index (NDVI) file
    :param output_csv_path: Path to the output CSV file
    :return:
    """

    factor_df = pd.read_csv(factor_file_path)
    index_df = pd.read_csv(index_file_path)

    # Convert coordinates to float and split them into 2 columns
    factor_df[['X', 'Y']] = factor_df['Coordinates'].str.split(',', expand=True).astype(float)
    index_df[['X', 'Y']] = index_df['Coordinates'].str.split(',', expand=True).astype(float)

    # Ensure the date column is in datetime format for consistency
    factor_df['Date'] = pd.to_datetime(factor_df['Date'])
    index_df['Date'] = pd.to_datetime(index_df['Date'])

    matched_df_values = []

    # Iterate through unique dates in the index dataset
    for date in index_df['Date'].unique():
        # Get the subset of index and factor data for the current date
        index_subset = index_df[index_df['Date'] == date]
        factor_subset = factor_df[factor_df['Date'] == date]

        if factor_subset.empty:
            print(f"Warning: No matching data found for date {date}. Filling with NaNs.")
            matched_df_values.extend([None] * len(index_subset))
            continue

        # Build KDTree for factor file coordinates for the current date
        factor_tree = cKDTree(factor_subset[['X', 'Y']].values)

        # Find the nearest factor coordinate for each index coordinate
        distances, indices = factor_tree.query(index_subset[['X', 'Y']].values)

        # Get matched values for the current date
        matched_df_values.extend(factor_subset.iloc[indices]['Value'].values)

    # Update values
    new_factor_df = index_df.copy()
    new_factor_df['Value'] = matched_df_values
    new_factor_df.drop(columns=['X', 'Y'], inplace=True)

    # Save update factor
    new_factor_df.to_csv(output_csv_path, index=False)
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
# # Change structure of file to match sentinel-2 file
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
# change_sentinel5_csv_structure("Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex/Amstel_AirQualityIndex_csv.csv",
#  "Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
#  "Datasets/Sentinel-5P/Amstelpark/Amstel_AirQualityIndex/Amstel_AirQualityIndex_csv.csv")
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
# Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# #Change structure of file to match sentinel-2 file
# change_csv_structure("Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv",
#  "Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
#  "Datasets/Sentinel-5P/Rembrandtpark/Rembrandt_AirQualityIndex/Rembrandt_AirQualityIndex_csv.csv")
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
# change_csv_structure("Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex/Wester_AirQualityIndex_csv.csv",
#                                "Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
#                                "Datasets/Sentinel-5P/Westerpark/Wester_AirQualityIndex/Wester_AirQualityIndex_csv.csv")
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
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-1/Vondelpark/Vondel_SoilMoisture_csv.csv",
#                      "Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv",
#                      "Datasets/Sentinel-1/Vondelpark/Vondel_SoilMoisture_csv.csv")
#
# ## AMSTEL
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Amstelpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv",
#                      "Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
#                      "Datasets/Sentinel-1/Amstelpark/Amstel_SoilMoisture_csv.csv")

# ## REMBRANDT
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Rembrandtpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv",
#                      "Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
#                      "Datasets/Sentinel-1/Rembrandtpark/Rembrandt_SoilMoisture_csv.csv")
#
# ## WESTER
#
# Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv"
# output_directory = "Datasets/Sentinel-1/Westerpark"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv",
#                      "Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
#                      "Datasets/Sentinel-1/Westerpark/Wester_SoilMoisture_csv.csv")
#
# #########################################################
# # Data Processing for Land Surface Temp (Sentinel-3) Files
# #########################################################
#
## VONDEL
#
# # Rename files
# output_directory = "Datasets/Sentinel-3/Vondelpark"
# new_prefix = "Vondel_LandTemp"
# rename_files(output_directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-3/Vondelpark/Vondel_LandTemp_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-3/Vondelpark/Vondel_LandTemp_csv.csv",
#                      "Datasets/Sentinel-2/Vondel_NDVI/Vondel_NDVI_csv.csv",
#               "Datasets/Sentinel-3/Vondelpark/Vondel_LandTemp_csv.csv")
#
## AMSTEL
#
# Rename files
# output_directory = "Datasets/Sentinel-3/Amstelpark"
# new_prefix = "Amstel_LandTemp"
# rename_files(output_directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-3/Amstelpark/Amstel_LandTemp_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-3/Amstelpark/Amstel_LandTemp_csv.csv",
#                      "Datasets/Sentinel-2/Amstel_NDVI/Amstel_NDVI_csv.csv",
#                      "Datasets/Sentinel-3/Amstelpark/Amstel_LandTemp_csv.csv")
#
## WESTER
#
# Rename files
# output_directory = "Datasets/Sentinel-3/Westerpark"
# new_prefix = "Wester_LandTemp"
# rename_files(output_directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-3/Westerpark/Wester_LandTemp_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-3/Westerpark/Wester_LandTemp_csv.csv",
#                      "Datasets/Sentinel-2/Wester_NDVI/Wester_NDVI_csv.csv",
#                      "Datasets/Sentinel-3/Westerpark/Wester_LandTemp_csv.csv")
#
## REMBRANDT
#
# Rename files
# output_directory = "Datasets/Sentinel-3/Rembrandtpark"
# new_prefix = "Rembrandt_LandTemp"
# rename_files(output_directory, old_prefix, new_prefix)
#
# # Convert GeoTIFF files to .csv file
# output_csv = "Datasets/Sentinel-3/Rembrandtpark/Rembrandt_LandTemp_csv.csv"
# convert_geotiff_to_csv(output_directory, output_csv)
#
# # Change csv file structure to match NDVI file
# change_csv_structure("Datasets/Sentinel-3/Rembrandtpark/Rembrandt_LandTemp_csv.csv",
#                      "Datasets/Sentinel-2/Rembrandt_NDVI/Rembrandt_NDVI_csv.csv",
#                      "Datasets/Sentinel-3/Rembrandtpark/Rembrandt_LandTemp_csv.csv")