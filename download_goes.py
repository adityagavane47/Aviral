import os
import urllib.request
import xml.etree.ElementTree as ET

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# We list the bucket for a specific day to find two real consecutive files
# GOES-16 ABI L2 Cloud and Moisture Imagery Full Disk (CMIPF)
PREFIX = "ABI-L2-CMIPF/2023/200/12/OR_ABI-L2-CMIPF-M6C13"
LIST_URL = f"https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix={PREFIX}"

print(f"Listing bucket to find files...\nURL: {LIST_URL}")

try:
    with urllib.request.urlopen(LIST_URL) as response:
        xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    keys = [elem.text for elem in root.findall('.//s3:Key', namespace)]
    
    if len(keys) < 2:
        print("Not enough files found.")
        exit(1)
        
    T0_URL = "https://noaa-goes16.s3.amazonaws.com/" + keys[0]
    T30_URL = "https://noaa-goes16.s3.amazonaws.com/" + keys[1]
    
    print(f"Found T0: {T0_URL}")
    print(f"Found T30: {T30_URL}")

    T0_FILE = os.path.join(DATA_DIR, "insat_t0.nc")
    T30_FILE = os.path.join(DATA_DIR, "insat_t30.nc")

    def download_file(url, dest):
        print(f"Downloading to {dest}...")
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest) / (1024 * 1024)
        print(f"[OK] Success! Size: {size:.2f} MB")

    download_file(T0_URL, T0_FILE)
    download_file(T30_URL, T30_FILE)
    print("Done! You can now run the pipeline.")

except Exception as e:
    print(f"[ERROR] {e}")
