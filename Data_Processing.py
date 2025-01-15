import os
import rasterio
import pandas as pd


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


# Update the `directory` path to the folder containing your files
directory = "Datasets/Sentinel-2/Sentinel2_Vondel"
# old_prefix = "openEO"
# new_prefix = "NO2_Vondel"
# rename_files(directory, old_prefix, new_prefix)

# Convert GeoTIFF files to .csv file
output_csv = "Datasets/Sentinel-2/Sentinel2_Vondel/Sentinel2_Vondel_CSV.csv"
convert_geotiff_to_csv(directory, output_csv)
