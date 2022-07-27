import numpy as np
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import h5py
filename = "logs/2022-02-25 105757_optimizer_log.hdf5"


with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])


print(data)