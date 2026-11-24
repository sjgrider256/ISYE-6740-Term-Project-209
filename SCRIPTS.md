# Scripts

## How to run

```
python Run.py
```

It prompts for a NEON site ID, validates it against `NEON_terrestrial_site_IDs.txt`,
and exits with an error if the ID is not in that list. The site is passed to the
three scripts through the `NEON_SITE` environment variable, and they run in order.
Any script that fails stops the pipeline.

Each script also runs standalone and falls back to `ORNL` if `NEON_SITE` is unset.

After the site, `Run.py` asks for a NEON API token and saves it to
`data/neon_token.txt` (gitignored), so you only type it once. It is passed to the
scripts as `NEON_TOKEN`. **A token is required** — as of June 2026 `neonutilities`
calls `auth_check()` before any transfer and raises `ConnectionError` without one.
Get a token from your user account at neonscience.org.

Fixed parameters: AOP year **2018**, vegetation structure **RELEASE-2024**. A site
with no data for those will error out rather than silently use a different year.

---

## 1. `download_neon_reflectance_data_and_chm_tiles.py`

Works out which 1 km AOP tiles the site's field plots fall in, then downloads them.

- Reads `data/All_NEON_TOS_Plot_Centroids_V11.shp` and keeps rows where
  `siteID` matches the site.
- Floors each plot's easting/northing to a 1000 m grid to get tile origins,
  then drops duplicates. Errors if the site has no plots.
- Loops the tile list twice, calling `neonutilities.by_tile_aop` once per tile so
  only one tile is in memory at a time:
  - `DP3.30006.001` hyperspectral reflectance → `data/refl/`
  - `DP3.30015.001` canopy height model → `data/chm/`
- Both loops use `skip_if_exists=True`, so re-running resumes instead of
  re-downloading.

**Output:** NEON's nested directory tree under `data/refl/` and `data/chm/`,
containing `*_reflectance.h5` and `*_CHM.tif` tiles.

**Note:** the repo currently only has the `.shp`; a shapefile also needs its
`.shx` and `.dbf` sidecars or `geopandas.read_file` cannot open it.

---

## 2. `NEON_TOS_Vegetation_Structure_data_wrangler.py`

Turns the TOS field survey into per-tree points with a genus label.

Part 1 — coordinates:
- Loads `DP1.10098.001` (vegetation structure), RELEASE-2024.
- Takes `vst_mappingandtagging` and drops rows with a blank `pointID`.
- For each unique `namedLocation.pointID`, hits the NEON locations API
  (1 s sleep per call to stay under the rate limit) for the reference point's
  UTM easting/northing and uncertainties. Points missing from the location
  database are dropped.
- Converts each stem's `stemDistance` / `stemAzimuth` offset from its reference
  point into `adjEasting` / `adjNorthing`.

Part 2 — stem heights:
- Loads `vst_apparentindividual`, keeps the most recent survey event per
  individual, keeps only `plantStatus == "Live"` with a positive height, and
  keeps the tallest stem per `individualID`.
- Inner-joins that onto the mapped coordinates.
- Derives `genus` as the first word of `scientificName` and keeps the 15 most
  abundant genera (`top_amount`).

**Output:**
- `data/veg_structure_data_by_genus_<SITE>.csv`
- `data/tree_genus_map_<SITE>.png` — map-view scatter coloured by genus

---

## 3. `tree_genus_classification.py`

Extracts a spectrum per tree and compares three classifiers.

- Reads the site CRS off the first CHM GeoTIFF with `rasterio` rather than
  hard coding an EPSG, so the field coordinates and raster grids always agree.
- Loads the stage 2 CSV into a GeoDataFrame in that CRS and assigns each tree to
  its 1 km tile, then matches tiles to downloaded 2018 HDF5 files.
- **CHM filter:** samples the CHM at each tree's location and drops stems more
  than `height_tolerance` (5 m) below canopy height — i.e. understory stems whose
  crown is not what the sensor sees. Trees with no CHM coverage are kept.
- `aop_h5refl2xarray()` reads a reflectance HDF5 into an `xarray.Dataset`
  (copied from the NEON pyXarray tutorial). It also flags bad bands: the two
  water-absorption windows from the file metadata plus the last 10 bands.
- For each tree, takes the single nearest pixel's full spectrum via
  `np.argmin` on the x/y coordinate arrays.
- Drops bad bands, does a stratified 80/20 split, `StandardScaler`, then
  `PCA(n_components=5)`.
- Fits Random Forest (200 trees), RBF SVM (C=10, gamma=0.1) and an MLP (32, 16),
  printing accuracy, macro OvO ROC AUC and a classification report for each.

**Output:** `data/NEON_<SITE>_Genus_Classification_Results.csv` — per-genus
precision/recall/F1/support plus pairwise OvO AUC, macro and weighted averages,
and an overall accuracy row, for all three models.

### Known quirk inherited from the NEON tutorial

In `aop_h5refl2xarray`, this line:

```python
refl_arrayT = np.transpose(refl_array, (1, 0, 2))
refl_arrayT = refl_array[::-1, :, :]
```

The transpose is immediately discarded by the second assignment. This matches
the upstream tutorial and has been left as-is; the y coordinates are built
ascending to match the flipped array, so it is self-consistent.

---

## Dependencies

```
numpy pandas geopandas rasterio h5py xarray scikit-learn
neonutilities matplotlib seaborn requests
```
