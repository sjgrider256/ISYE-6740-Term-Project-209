import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = ["download_neon_reflectance_data_and_chm_tiles.py",
           "NEON_TOS_Vegetation_Structure_data_wrangler.py",
           "tree_genus_classification.py"]

sites = open("NEON_terrestrial_site_IDs.txt").read().split()

# Check for existing AOP tiles (site ID, easting, northing)
refl_tiles = {
    (p.name.split("_")[2], *p.name.split("_")[-3:-1])
    for p in Path("data/refl").rglob("NEON_*_reflectance.h5")
    if "2018" in p.parts
}

chm_tiles = {
    (p.name.split("_")[2], *p.name.split("_")[-3:-1])
    for p in Path("data/chm").rglob("NEON_*_CHM.tif")
    if "2018" in p.parts
}

matching_tiles = refl_tiles & chm_tiles
matching_sites = {tile[0] for tile in matching_tiles}

if len(matching_sites) > 1:
    sys.exit("ERROR: Matching AOP tiles found for multiple sites.")

elif matching_tiles:
    site_ID = next(iter(matching_sites))
    print(f"Detected NEON site: {site_ID}")
    print(f"Found {len(matching_tiles)} matching AOP tile pairs.")
    print("Skipping AOP download.")
    SCRIPTS = SCRIPTS[1:]

else:
    site_ID = (input("Enter NEON site ID [ORNL]: ").strip().upper() or "ORNL")

if site_ID not in sites:
    sys.exit(f"ERROR: '{site_ID}' is not a NEON terrestrial site ID. See NEON_terrestrial_site_IDs.txt")

os.environ["NEON_SITE"] = site_ID

# NEON API token: reused from data/neon_token.txt if saved, otherwise prompted once.
# Required - as of June 2026 NEON rejects downloads without a token.
token_file = "data/neon_token.txt"
if os.path.exists(token_file):
    token = open(token_file).read().strip()
    print(f"Using NEON token from {token_file}")
else:
    token = input("Enter NEON API token: ").strip()
    if token:
        open(token_file, "w").write(token)
        print(f"Saved to {token_file}")

if not token:
    sys.exit("ERROR: a NEON API token is required to download data. Get one from your user account at neonscience.org")

os.environ["NEON_TOKEN"] = token

for script in SCRIPTS:
    print(f"\n===== {script}  (site {site_ID}) =====")
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"ERROR: {script} failed. Stopping.")

print(f"\nDone. Results: data/NEON_{site_ID}_Genus_Classification_Results.csv")
