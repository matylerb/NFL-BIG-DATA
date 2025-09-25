import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#Path
DATA_PATH = 'dataset/' 

games = pd.read_csv(f'{DATA_PATH}games.csv')
plays = pd.read_csv(f'{DATA_PATH}plays.csv')
players = pd.read_csv(f'{DATA_PATH}players.csv')
pff_scouting = pd.read_csv(f'{DATA_PATH}pffScoutingData.csv')