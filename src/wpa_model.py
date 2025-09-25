import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_PATH = 'dataset/' 
LB_POSITIONS = ['LB', 'ILB', 'OLB', 'MLB'] 

def load_data(data_path):
    """Loads necessary CSV files."""
    print("Loading core data files...")

    try:
        games = pd.read_csv(os.path.join(data_path, 'games.csv'))
        plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))
        players = pd.read_csv(os.path.join(data_path, 'players.csv'))
        pff_scouting = pd.read_csv(os.path.join(data_path, 'pffScoutingData.csv'))
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required file. Check your DATA_PATH: {e}")
        raise
        
    return games, plays, players, pff_scouting

def prepare_data(games, plays, players, pff_scouting, LB_POSITIONS):
    """Merges data and aggregates LB performance metrics."""

    # 1. Base Play-by-Play Log (PBP)
    pbp_df = plays.merge(games, on='gameId', how='left')

    # 2. Identify Linebackers (LBs)
    print("Identifying Linebackers...")
    lbs = players[players['officialPosition'].isin(LB_POSITIONS)][['nflId', 'displayName']].copy()

    # 3. Extract and Aggregate LB Performance
    print("Aggregating Linebacker performance...")
    
    pff_scouting = pff_scouting.dropna(subset=['nflId'])
    lb_pff = pff_scouting.merge(lbs, on='nflId', how='inner')

    # Aggregate LB metrics (sacks, hurries, hits) for each unique play
    lb_performance_agg = lb_pff.groupby(['gameId', 'playId']).agg(
        def_lb_sacks=('pff_sack', 'sum'),
        def_lb_hurries=('pff_hurry', 'sum'),
        def_lb_hits=('pff_hit', 'sum')
    ).reset_index()

    # 4. Final Merge & Cleanup
    pbp_df = pbp_df.merge(lb_performance_agg, on=['gameId', 'playId'], how='left')

    # Fill NaN for plays where no LB recorded a sack, hit, or hurry (assume 0)
    pbp_df[['def_lb_sacks', 'def_lb_hurries', 'def_lb_hits']] = pbp_df[['def_lb_sacks', 'def_lb_hurries', 'def_lb_hits']].fillna(0)
    
    # Drop rows that are essential for WPA calculation but missing data
    pbp_df = pbp_df.dropna(subset=['down', 'yardsToGo', 'yardlineNumber', 'gameClock', 'possessionTeam', 'homeTeamAbbr']).copy()

    return pbp_df, lb_pff

