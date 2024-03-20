import json

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

with open('config', "r") as f_json:
    config = json.loads(f_json.read())