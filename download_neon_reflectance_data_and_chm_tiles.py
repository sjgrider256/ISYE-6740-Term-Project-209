# Required libraries
import numpy as np
import geopandas as gpd
import neonutilities as nu

#load NEON Terrestrial Observation System plot centroid shapefile
tos_centroids = gpd.read_file("data/All_NEON_TOS_Plot_Centroids_V11.shp")

#specify site
site_ID = "ORNL"

#select site
site_plots = tos_centroids[tos_centroids['siteID'] == site_ID].copy()

#transform site extent for matching
site_plots['tile_easting'] = (np.floor(site_plots['easting'] / 1000) * 1000).astype(int)
site_plots['tile_northing'] = (np.floor(site_plots['northing'] / 1000) * 1000).astype(int)

#drop duplicate tiles
site_tiles = (site_plots[['tile_easting', 'tile_northing']].drop_duplicates().sort_values(['tile_easting', 'tile_northing']).reset_index(drop=True))

#to avoid memory limits, loop through each tile, extract AOP data, and discard from memory before proceeding to the next tile

#reflectance
for i in range(len(site_tiles)):

    tile_easting = int(site_tiles.iloc[i]['tile_easting'])

    tile_northing = int(site_tiles.iloc[i]['tile_northing'])

    print("Downloading tile", i + 1, "of", len(site_tiles), ":", tile_easting, tile_northing)

    nu.by_tile_aop(
        dpid='DP3.30006.001',
        site=site_id,
        year=2018,
        easting=tile_easting,
        northing=tile_northing,
        include_provisional=False,
        check_size=False,
        token=NEON_TOKEN,
        savepath='./data/refl'
    )

#Canopy Height Model (CHM) data
for i in range(len(site_tiles)):
    tile_easting = int(site_tiles.iloc[i]['tile_easting'])
    tile_northing = int(site_tiles.iloc[i]['tile_northing'])

    nu.by_tile_aop(
        dpid='DP3.30015.001',
        site=site_id,
        year=2018,
        easting=tile_easting,
        northing=tile_northing,
        include_provisional=False,
        check_size=False,
        token=NEON_TOKEN,
        savepath='./data/chm',
        skip_if_exists=True
    )



