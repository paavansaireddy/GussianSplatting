import os
import requests
import subprocess

# Define the directory where the repo should be cloned
directory = '/content/gaussian-splatting'

# Define the URL of the new train.py
url = 'https://raw.githubusercontent.com/paavansaireddy/3DGaussianSplatting/main/train.py'

# Check if the directory exists, if not, clone the repository
if not os.path.exists(directory):
    # Clone the repository
    subprocess.run(['git', 'clone', '--recursive', 'https://github.com/camenduru/gaussian-splatting', directory])
    print(f"Repository cloned into {directory}")
else:
    print(f"Directory {directory} already exists.")

# Change to the directory
os.chdir(directory)

# Remove the existing train.py file if it exists
if os.path.exists("train.py"):
    os.remove("train.py")
    print("Existing train.py removed.")

# Download the new train.py from the given URL
response = requests.get(url)
if response.status_code == 200:
    with open("train.py", "wb") as f:
        f.write(response.content)
    print("train.py has been updated.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
