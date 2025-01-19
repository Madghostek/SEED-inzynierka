from pathlib import Path
import os


if "PLG_GROUPS_STORAGE" in os.environ:
    storage = Path("/net/tscratch/people/plgtsroka/") # scratch is faster 
else:
    storage = Path("/home/tomek/")
base_path=storage/"datasets" # creates a subdirectory there
meta_fname="meta.json"  # name of the dataset metadata file
