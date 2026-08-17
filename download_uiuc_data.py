import urllib.request
import zipfile
import os

url = "https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip"
zip_path = "coord_seligfmt.zip"
extract_dir = "uiuc_airfoils"

urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("Extracted to:", extract_dir)
print("Top-level contents:", os.listdir(extract_dir)[:10])