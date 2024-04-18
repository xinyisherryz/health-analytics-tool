import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib as plt
import joblib
from sklearn import __version__ as skl_version

print("Streamlit version:", st.__version__)
print("Pandas version:", pd.__version__)
print("Seaborn version:", sns.__version__)
print("Matplotlib version:", plt.__version__)
print("Joblib version:", joblib.__version__)
print("Scikit-learn version:", skl_version)
