import os
import pandas as pd

location = os.path.dirname(os.path.realpath(__file__))
file = os.path.join(location, '/src/cypack/data', 'Data.csv')

Data=pd.read_csv(file,sep=";")
