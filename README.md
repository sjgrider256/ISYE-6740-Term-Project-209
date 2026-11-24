# ISYE-6740-Term-Project-209
### Individual tree species classification using hyperspectral NEON data
Neon AOP & TOS Data Prep Guide: https://www.neonscience.org/resources/learning-hub/tutorials/refl-classification-pyxarray

This project explores the use of machine learning for forest species classification using data from the National Ecological Observatory Network (NEON). Field-observed tree species from NEON's Terrestrial Observation System (TOS) are combined with hyperspectral imagery and LiDAR from the Airborne Observation Platform (AOP) to characterize individual canopy trees.

The workflow includes spatially matching field observations to tree crowns, extracting hyperspectral and LiDAR features, reducing spectral dimensionality with PCA, and classifying tree species by comparing SVM, Random Forest, and neural-network model performace. Performance will be evaluated using spatially grouped validation and a held-out forest plot to test how well the models generalize to unseen areas. Data extraction and preparation methodologies were adopted from the following the studies. 

### Literature: 
https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002700

https://www.sciencedirect.com/science/article/pii/S0034425722003704

## Quick Start

### 1. Install dependencies

Clone or download this repository, open a terminal in the repository's root directory, and install the required Python packages:

```bash
pip install numpy pandas geopandas rasterio h5py xarray scikit-learn neonutilities matplotlib seaborn requests
```

### 2. Launch the pipeline

From the repository's root directory, run:

```bash
python Run.py
```

When prompted:

- **NEON Site ID:** Enter a valid terrestrial NEON site ID (e.g., `ORNL`). Press Enter to use ORNL by default. Supported sites are listed in `NEON_terrestrial_site_IDs.txt`.
- **NEON API Token:** Enter your API token, available through your user account at [neonscience.org](https://www.neonscience.org/). Your token is saved locally and reused in subsequent runs.

The pipeline automatically downloads the required imagery and vegetation structure data, processes tree locations, extracts hyperspectral signatures, and trains and evaluates three machine learning classifiers.

### 3. Using existing AOP imagery (optional)

If you already have NEON reflectance and canopy height model (CHM) tiles from 2018, place them in the following directories:

```text
data/
├── refl/
│   └── 2018/
│       └── NEON_D07_ORNL_DP3_744000_3983000_reflectance.h5
└── chm/
    └── 2018/
        └── NEON_D07_ORNL_DP3_744000_3983000_CHM.tif
```

The filenames above are an example using the ORNL observation tower tile.

The pipeline automatically detects matching tiles, identifies the site, and skips imagery downloads. A single matching tile pair is sufficient to initiate the pipeline, provided enough field observations exist within its extent.

- Reflectance and CHM tiles must have matching site IDs and tile coordinates.
- Only one site with matching imagery may be present at a time.
- An API token is still required to retrieve vegetation structure data.

When switching sites, move existing imagery outside these directories.

### 4. View results

All outputs are saved to the `data/` directory:

| Output | Description |
|---|---|
| `veg_structure_data_by_genus_<SITE>.csv` | Geolocated tree inventory |
| `tree_genus_map_<SITE>.png` | Spatial distribution of tree genera |
| `NEON_<SITE>_Genus_Classification_Results.csv` | Classification performance metrics |

### Important Notes

- The pipeline uses 2018 NEON AOP imagery and RELEASE-2024 vegetation structure data.
- The `All_NEON_TOS_Plot_Centroids_V11` shapefile and its supporting files must remain in the `data/` directory.
- Classification requires at least three genera with a minimum of three samples per genus. Sites with insufficient observations may not support successful classification.
- Downloaded hyperspectral imagery can require substantial disk space.

For a detailed description of each processing stage, see [SCRIPTS.md](SCRIPTS.md).
