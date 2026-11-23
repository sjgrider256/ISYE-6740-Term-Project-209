# Data processing and geospatial
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import h5py
import xarray as xr

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

site_crs = 'EPSG:32616'

trees_df = pd.read_csv('data/veg_structure_data_by_genus.csv')

trees_gdf = gpd.GeoDataFrame(trees_df, geometry=gpd.points_from_xy(trees_df['adjEasting'], trees_df['adjNorthing']), crs=site_crs)

trees_gdf['pixel_easting'] = trees_gdf.geometry.x
trees_gdf['pixel_northing'] = trees_gdf.geometry.y
trees_gdf['height'] = pd.to_numeric(trees_gdf['height'], errors='coerce')

# Remove trees with missing or invalid coordinates
trees_gdf = trees_gdf[np.isfinite(trees_gdf["pixel_easting"]) & np.isfinite(trees_gdf["pixel_northing"])].copy()

# Load NEON data
refl_folder = Path("data/refl")
h5_files_2018 = []
for file in refl_folder.rglob("*_reflectance.h5"):
    if "2018" in file.parts:
        h5_files_2018.append(file)
h5_files_2018.sort()

tile_files = []
for file in h5_files_2018:
    filename = file.name
    parts = filename.split("_")
    tile_easting = int(parts[-3])
    tile_northing = int(parts[-2])
    record = {"tile_easting": tile_easting, "tile_northing": tile_northing, "file": file}
    tile_files.append(record)
tile_files_df = pd.DataFrame(tile_files)

#assign each endmember to its 1 km AOP tile

trees_gdf["tile_easting"] = (np.floor(trees_gdf["pixel_easting"] / 1000) * 1000).astype(int)
trees_gdf["tile_northing"] = (np.floor(trees_gdf["pixel_northing"] / 1000) * 1000).astype(int)

#unique AOP tiles containing hemlocks


tree_tiles = trees_gdf[
    ["tile_easting", "tile_northing"]
].drop_duplicates().sort_values(
    ["tile_easting", "tile_northing"]
).reset_index(drop=True)

print("Number of trees", len(trees_gdf))
print("Number of AOP tiles containing endmembers:", len(tree_tiles))

# Match tree-containing tiles to downloaded HDF5 files
tree_tile_files = tree_tiles.merge(
    tile_files_df,
    on=['tile_easting', 'tile_northing'],
    how='left'
)
tree_tile_files = tree_tile_files.dropna()
print("Tree tile files:", len(tree_tile_files))

#filter branches under canopy

chm_files = list(Path('./data/chm').rglob('*_CHM.tif'))
trees_gdf['chm_height'] = np.nan

total_tiles = len(tree_tile_files)

for i, (_, tile) in enumerate(tree_tile_files.iterrows(), start=1):
    e = int(tile['tile_easting'])
    n = int(tile['tile_northing'])
    matches = [p for p in chm_files if p.name.endswith(f'_{e}_{n}_CHM.tif')]

    if len(matches) != 1:
        print(f'Missing CHM for tile {e}, {n} — skipping')
        continue
    mask = (trees_gdf['tile_easting'] == e) & (trees_gdf['tile_northing'] == n)

    with rasterio.open(matches[0]) as src:
        for idx, tree in trees_gdf.loc[mask].iterrows():
            value = next(src.sample([(tree.geometry.x, tree.geometry.y)], masked=True))[0]

            if not np.ma.is_masked(value) and np.isfinite(value):
                trees_gdf.at[idx, 'chm_height'] = float(value)

# Filter understory stems only where CHM data exists
height_tolerance = 5.0 #meters
original_count = len(trees_gdf)
has_chm = trees_gdf['chm_height'].notna()

trees_gdf = trees_gdf[
    ~has_chm |
    (trees_gdf['height'] >= trees_gdf['chm_height'] - height_tolerance)
].copy()

print(f"Original stems: {original_count}")
print(f"Retained stems: {len(trees_gdf)}")
print(f"Stems without CHM data: {trees_gdf['chm_height'].isna().sum()}")

#this script loads NEON AOP Reflectance tiles as Xarray
#copied from  https://www.neonscience.org/resources/learning-hub/tutorials/refl-classification-pyxarray
#Xarrays make multidimensional data processing more efficent: https://docs.xarray.dev/en/stable/

def aop_h5refl2xarray(h5_filename):
    """
    Reads a NEON AOP reflectance HDF5 file and returns an xarray.Dataset with reflectance and weather quality indicator data.

    Parameters
    ----------
    h5_filename : str
        Path to the NEON AOP reflectance HDF5 file.

    Returns
    -------
    dsT : xarray.Dataset
        An xarray Dataset containing:
            - 'reflectance': DataArray of reflectance values (y, x, wavelengths)
            - 'weather_quality_indicator': DataArray of weather quality indicator (y, x)
            - Coordinates: y (UTM northing), x (UTM easting), wavelengths, fwhm, good_wavelengths
            - Metadata attributes: projection, spatial_ref, EPSG, no_data_value, scale_factor, bad_band_window1, bad_band_window2, etc.
    """

    with h5py.File(h5_filename) as hdf5_file:
        print('Reading in ', h5_filename)
        sitename = list(hdf5_file.keys())[0]
        h5_refl_group = hdf5_file[sitename]['Reflectance']
        refl_dataset = h5_refl_group['Reflectance_Data']
        refl_array = refl_dataset[()].astype('float32')

        # Transpose and flip reflectance data
        refl_arrayT = np.transpose(refl_array, (1, 0, 2))
        refl_arrayT = refl_array[::-1, :, :]

        refl_shape = refl_arrayT.shape
        wavelengths = h5_refl_group['Metadata']['Spectral_Data']['Wavelength'][:]
        fwhm = h5_refl_group['Metadata']['Spectral_Data']['FWHM'][:]

        # Weather Quality Indicator: transpose and flip to match reflectance
        wqi_array = h5_refl_group['Metadata']['Ancillary_Imagery']['Weather_Quality_Indicator'][()]
        wqi_arrayT = np.transpose(wqi_array, (1, 0, 2))
        wqi_arrayT = wqi_array[::-1, :, :]

        # Collect metadata
        metadata = {}
        metadata['shape'] = refl_shape
        metadata['no_data_value'] = float(refl_dataset.attrs['Data_Ignore_Value'])
        metadata['scale_factor'] = float(refl_dataset.attrs['Scale_Factor'])
        metadata['bad_band_window1'] = h5_refl_group.attrs['Band_Window_1_Nanometers']
        metadata['bad_band_window2'] = h5_refl_group.attrs['Band_Window_2_Nanometers']
        metadata['projection'] = h5_refl_group['Metadata']['Coordinate_System']['Proj4'][()].decode('utf-8')
        metadata['spatial_ref'] = h5_refl_group['Metadata']['Coordinate_System']['Coordinate_System_String'][()].decode('utf-8')
        metadata['EPSG'] = int(h5_refl_group['Metadata']['Coordinate_System']['EPSG Code'][()])

        # Parse map info for georeferencing
        map_info = str(h5_refl_group['Metadata']['Coordinate_System']['Map_Info'][()]).split(",")
        pixel_width = float(map_info[5])
        pixel_height = float(map_info[6])
        x_min = float(map_info[3]); x_min = int(x_min)
        y_max = float(map_info[4]); y_max = int(y_max)
        x_max = x_min + (refl_shape[1]*pixel_width); x_max = int(x_max)
        y_min = y_max - (refl_shape[0]*pixel_height); y_min = int(y_min)

        # Calculate UTM coordinates for x and y axes
        x_coords = np.linspace(x_min, x_max, num=refl_shape[1]).astype(float)
        y_coordsT = np.linspace(y_min, y_max, num=refl_shape[0]).astype(float)

        # Flag good/bad wavelengths (1=good, 0=bad)
        good_wavelengths = np.ones_like(wavelengths)
        for bad_window in [metadata['bad_band_window1'], metadata['bad_band_window2']]:
            bad_indices = np.where((wavelengths >= bad_window[0]) & (wavelengths <= bad_window[1]))[0]
            good_wavelengths[bad_indices] = 0
        good_wavelengths[-10:] = 0

        # Create xarray DataArray for reflectance
        refl_xrT = xr.DataArray(
            refl_arrayT,
            dims=["y", "x", "wavelengths"],
            name="reflectance",
            coords={
                "y": ("y", y_coordsT),
                "x": ("x", x_coords),
                "wavelengths": ("wavelengths", wavelengths),
                "fwhm": ("wavelengths", fwhm),
                "good_wavelengths": ("wavelengths", good_wavelengths)
            }
        )

        # Create xarray DataArray for Weather Quality Indicator
        wqi_xrT = xr.DataArray(
            wqi_arrayT,
            dims=["y", "x", "wqi_channel"],
            name="weather_quality_indicator",
            coords={
                "y": ("y", y_coordsT),
                "x": ("x", x_coords),
                "wqi_channel": ("wqi_channel", [0, 1, 2])
            }
        )

        # Create xarray Dataset and add metadata as attributes
        dsT = xr.Dataset({
            "reflectance": refl_xrT,
            "weather_quality_indicator": wqi_xrT
        })
        for key, value in metadata.items():
            if key not in ['shape', 'extent', 'ext_dict']:
                dsT.attrs[key] = value

        return dsT
      
# match NEON pixels to tree coordinates and extract spectra
spectra = []
pixel_metadata = []

total_tiles = len(tree_tile_files)

for i, (_, tile) in enumerate(tree_tile_files.iterrows(), start=1):
    print(f'Processing reflectance tile {i}/{total_tiles}')

    targets_inside = trees_gdf[
        (trees_gdf['tile_easting'] == tile['tile_easting']) &
        (trees_gdf['tile_northing'] == tile['tile_northing'])
    ]

    if targets_inside.empty:
        continue

    neon_tile = aop_h5refl2xarray(tile['file'])['reflectance']
    x_coords = neon_tile.x.values
    y_coords = neon_tile.y.values

    for _, tree in targets_inside.iterrows():
        col = np.argmin(np.abs(x_coords - tree['pixel_easting']))
        row = np.argmin(np.abs(y_coords - tree['pixel_northing']))

        # Extract only the nearest pixel
        spectrum = neon_tile.isel(y=row, x=col).values

        spectra.append(spectrum)
        pixel_metadata.append({
            'individualID': tree['individualID'],
            'genus': tree['genus'],
            'scientificName': tree['scientificName'],
            'height': tree['height'],
            'chm_height': tree['chm_height']
        })

spectra = np.asarray(spectra)
pixel_metadata = pd.DataFrame(pixel_metadata)

print("Extracted spectra shape:", spectra.shape)
print("Unique trees:", pixel_metadata['individualID'].nunique())

#assign dependent and independent variables
X = np.asarray(spectra)
y = pixel_metadata['genus'].to_numpy()
print("\nSamples per genus:")
print(pd.Series(y).value_counts())

#filter bad bands
good_bands = neon_tile.coords["good_wavelengths"].values == 1
X = X[:, good_bands]
filtered_wavelengths = neon_tile.coords["wavelengths"].values[good_bands]
print("Remaining bands:", X.shape[1])

#80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=20
)

print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))

#scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("\nExplained variance per component:")
for i, variance in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"PC{i}: {variance:.2%}")
print(f"\nTotal explained variance: {pca.explained_variance_ratio_.sum():.2%}")

#Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=20)
rf.fit(X_train_pca, y_train)
y_pred = rf.predict(X_test_pca)
y_prob = rf.predict_proba(X_test_pca)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Multiclass ROC AUC:", roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro", labels=rf.classes_))
print(classification_report(y_test, y_pred, zero_division=0))

#RBF-SVM
svm = SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=20)
svm.fit(X_train_pca, y_train)
y_pred = svm.predict(X_test_pca)
y_prob = svm.predict_proba(X_test_pca)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Multiclass ROC AUC:", roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro", labels=svm.classes_))
print(classification_report(y_test, y_pred, zero_division=0))

#MLP Neural Network Classifier
nn = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=1000, random_state=20)
nn.fit(X_train_pca, y_train)
y_pred = nn.predict(X_test_pca)
y_prob = nn.predict_proba(X_test_pca)
print("Neural Network Accuracy:", accuracy_score(y_test, y_pred))
print("Multiclass ROC AUC:", roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro", labels=nn.classes_))
print(classification_report(y_test, y_pred, zero_division=0))

#export results as CSV
models = {"Random Forest": rf, "RBF SVM": svm, "Neural Network": nn}
results = []

for name, model in models.items():
    pred = model.predict(X_test_pca)
    prob = model.predict_proba(X_test_pca)
    report = classification_report(y_test, pred, labels=model.classes_, output_dict=True, zero_division=0)

    for i, genus in enumerate(model.classes_):
        pair_aucs = []

        for other in model.classes_:
            if genus == other: continue
            mask = np.isin(y_test, [genus, other])
            if np.unique(y_test[mask]).size < 2: continue
            pair_aucs.append(roc_auc_score(y_test[mask] == genus, prob[mask, i]))

        metrics = report[genus]
        results.append({"Model": name, "Genus": genus, "Precision": metrics["precision"], "Recall": metrics["recall"],
                        "F1": metrics["f1-score"], "Support": metrics["support"],
                        "ROC AUC (OvO)": np.mean(pair_aucs) if pair_aucs else np.nan})

    for label in ["macro avg", "weighted avg"]:
        metrics = report[label]
        results.append({"Model": name, "Genus": label, "Precision": metrics["precision"], "Recall": metrics["recall"],
                        "F1": metrics["f1-score"], "Support": metrics["support"]})

    overall_auc = roc_auc_score(y_test, prob, multi_class="ovo", average="macro", labels=model.classes_)

    results.append({"Model": name, "Genus": "Overall", "Support": len(y_test),
                    "Accuracy": accuracy_score(y_test, pred), "ROC AUC (OvO)": overall_auc})

results_df = pd.DataFrame(results)
results_df.to_csv("data/NEON_ORNL_Genus_Classification_Results.csv", index=False, float_format="%.4f")

print("Saved: data/NEON_ORNL_Genus_Classification_Results.csv")
