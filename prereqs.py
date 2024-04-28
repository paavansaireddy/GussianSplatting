import os
import requests
import subprocess
import shutil

# Define the directory where the repo should be cloned
directory = '/content/gaussian-splatting'

# Define the URL of the new train.py
url = 'https://github.com/paavansaireddy/GussianSplatting/blob/main/train.py'

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

source_file = "/content/GussianSplatting/train.py"
destination_file = "/content/gaussian-splatting"

# Copy the file
try:
    shutil.copy(source_file, destination_file)
    print(f"File copied successfully from {source_file} to {destination_file}")
except IOError as e:
    # Handle the error, e.g., source or destination doesn't exist
    print(f"Failed to copy file: {e}")
except Exception as e:
    # Handle other exceptions, such as permission issues
    print(f"An error occurred: {e}")

source_path1 = "/content/GussianSplatting/playroom"
destination_path1 = "/content/gaussian-splatting"

# Copy the file
try:
    shutil.move(source_path1, destination_path1)
    print(f"File moved successfully from {source_path1} to {destination_path1}")
except IOError as e:
    # Handle the error, e.g., source or destination doesn't exist
    print(f"Failed to move file: {e}")
except Exception as e:
    # Handle other exceptions, such as permission issues
    print(f"An error occurred: {e}")


