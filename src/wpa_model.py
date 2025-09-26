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
    # Using os.path.join for platform-independent path construction
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

    # Base Play-by-Play Log (PBP)
    pbp_df = plays.merge(games, on='gameId', how='left')

    # Identify Linebackers (LBs)
    print("Identifying Linebackers...")
    lbs = players[players['officialPosition'].isin(LB_POSITIONS)][['nflId', 'displayName']].copy()

    # Extract and Aggregate LB Performance
    print("Aggregating Linebacker performance...")
    
    pff_scouting = pff_scouting.dropna(subset=['nflId'])
    lb_pff = pff_scouting.merge(lbs, on='nflId', how='inner')

    # Aggregate LB metrics (sacks, hurries, hits) for each unique play
    lb_performance_agg = lb_pff.groupby(['gameId', 'playId']).agg(
        def_lb_sacks=('pff_sack', 'sum'),
        def_lb_hurries=('pff_hurry', 'sum'),
        def_lb_hits=('pff_hit', 'sum')
    ).reset_index()

    # Final Merge & Cleanup
    pbp_df = pbp_df.merge(lb_performance_agg, on=['gameId', 'playId'], how='left')

    # Fill NaN for plays where no LB recorded a sack, hit, or hurry (assume 0)
    pbp_df[['def_lb_sacks', 'def_lb_hurries', 'def_lb_hits']] = pbp_df[['def_lb_sacks', 'def_lb_hurries', 'def_lb_hits']].fillna(0)
    
    # Drop rows that are essential for WPA calculation but missing data
    pbp_df = pbp_df.dropna(subset=['down', 'yardsToGo', 'yardlineNumber', 'gameClock', 'possessionTeam', 'homeTeamAbbr']).copy()

    return pbp_df, lb_pff

def feature_engineer(df):
    """Creates WPA features and the target variable."""
    print("Engineering WPA features...")
    
    # Time Remaining (in seconds)
    df['gameClock_seconds'] = df['gameClock'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    df['time_remaining'] = 15 * 60 * (4 - df['quarter']) + df['gameClock_seconds']

    # Score Differential (Offense perspective)
    df['score_differential'] = np.where(
        df['possessionTeam'] == df['homeTeamAbbr'],
        df['preSnapHomeScore'] - df['preSnapVisitorScore'],
        df['preSnapVisitorScore'] - df['preSnapHomeScore']
    )

    # Field Position (normalized distance to opponent's endzone)
    df['yardline_possession'] = np.where(
        df['yardlineSide'] == df['possessionTeam'],
        df['yardlineNumber'],
        100 - df['yardlineNumber']
    )
    df['field_pos_norm'] = df['yardline_possession'] / 100 

    # Determine Final Winner (Assuming last score is the final score)
    df['FinalHomeScore'] = df.groupby('gameId')['preSnapHomeScore'].transform('last')
    df['FinalVisitorScore'] = df.groupby('gameId')['preSnapVisitorScore'].transform('last')
    
    df['winning_team'] = np.where(
        df['FinalHomeScore'] > df['FinalVisitorScore'], df['homeTeamAbbr'],
        np.where(df['FinalVisitorScore'] > df['FinalHomeScore'], df['visitorTeamAbbr'], 'TIE')
    )
    
    # Target Variable: Did the team with possession win the game?
    df['TeamInPossessionWin'] = np.where(
        df['possessionTeam'] == df['winning_team'], 1, 0
    )
    
    df = df[df['winning_team'] != 'TIE'].copy()
    
    return df

def train_and_predict_wp(df):
    """Trains the baseline WP model and predicts WP_Start for all plays."""
    print("Training baseline Win Probability model...")

    WP_FEATURES = ['down', 'yardsToGo', 'field_pos_norm', 'score_differential', 'time_remaining']

    X = df[WP_FEATURES]
    y = df['TeamInPossessionWin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the Logistic Regression Model
    wp_model = LogisticRegression(solver='lbfgs', max_iter=2000) 
    wp_model.fit(X_train, y_train)

    print(f"Baseline WP Model AUC Score (on test set): {roc_auc_score(y_test, wp_model.predict_proba(X_test)[:, 1]):.4f}")

    # Predict WP_Start for the entire dataset
    df['WP_Start'] = wp_model.predict_proba(df[WP_FEATURES])[:, 1]
    
    return df

def calculate_lb_wpa(pbp_df, lb_pff):
    """Calculates WPA for each play and attributes credit to LBs."""
    print("Calculating Win Probability Added (WPA)...")

    # 1. Calculate WP_End (WP_Start of the NEXT play)
    pbp_df['WP_End'] = pbp_df.groupby('gameId')['WP_Start'].shift(-1)
    
    # Handle the last play of the game: WP_End = Final Result (1.0 or 0.0)
    pbp_df['WP_End'] = np.where(pbp_df['WP_End'].isna(), pbp_df['TeamInPossessionWin'].astype(float), pbp_df['WP_End'])

    # Handle Turnover: If possession changes, WP must be flipped (1 - WP_End)
    possession_changed = (pbp_df.groupby('gameId')['possessionTeam'].shift(-1) != pbp_df['possessionTeam'])
    pbp_df['WP_End'] = np.where(
        (possession_changed) & (pbp_df['WP_End'] != 0.0) & (pbp_df['WP_End'] != 1.0),
        1 - pbp_df['WP_End'],
        pbp_df['WP_End']
    )
    
    # 2. Calculate WPA_Play
    pbp_df['WPA_Play'] = pbp_df['WP_End'] - pbp_df['WP_Start']
    
    # 3. Assign WPA Credit to Defensive LBs and get the defensive team
    lb_contributions = pbp_df[
        (pbp_df['def_lb_sacks'] > 0) | 
        (pbp_df['def_lb_hurries'] > 0) | 
        (pbp_df['def_lb_hits'] > 0)
    ][['gameId', 'playId', 'WPA_Play', 'defensiveTeam', 'winning_team', 'def_lb_sacks', 'def_lb_hurries', 'def_lb_hits']].copy()

    individual_lb_wpa = lb_pff.merge(lb_contributions, on=['gameId', 'playId'], how='inner')
    
    individual_lb_wpa['WPA_Credit'] = -individual_lb_wpa['WPA_Play']
    
    # Games Won
    game_outcomes = individual_lb_wpa[['nflId', 'displayName', 'gameId', 'defensiveTeam', 'winning_team']].drop_duplicates()
    game_outcomes['is_win'] = (game_outcomes['defensiveTeam'] == game_outcomes['winning_team']).astype(int)
    
    wins_per_player = game_outcomes.groupby(['nflId', 'displayName', 'defensiveTeam'])['is_win'].sum().reset_index()
    wins_per_player = wins_per_player.rename(columns={'is_win': 'Games_Won'})

    # 4. Calculate Final Cumulative LB WPA (Player Evaluation Metric)
    final_lb_wpa = individual_lb_wpa.groupby(['nflId', 'displayName', 'defensiveTeam']).agg(
        Total_WPA_Contribution=('WPA_Credit', 'sum'),
        Total_Plays_Contributed=('playId', 'count'),
        Sacks_Count=('def_lb_sacks', 'sum'),
        Hurries_Count=('def_lb_hurries', 'sum'),
        Hits_Count=('def_lb_hits', 'sum'),
        Game_Count=('gameId', pd.Series.nunique)
    ).reset_index()
    
    final_lb_wpa = final_lb_wpa.merge(wins_per_player, on=['nflId', 'displayName', 'defensiveTeam'], how='left')
    
    # Calculate WPA per Play (Efficiency Metric)
    final_lb_wpa['WPA_Per_Play'] = final_lb_wpa['Total_WPA_Contribution'] / final_lb_wpa['Total_Plays_Contributed']
    final_lb_wpa['Win_Rate'] = final_lb_wpa['Games_Won'] / final_lb_wpa['Game_Count']

    final_lb_wpa = final_lb_wpa.sort_values(by='Total_WPA_Contribution', ascending=False)
    
    return final_lb_wpa

#Save results
def save_results_to_csv(df, filename='top_100_linebackers.csv', num_rows=100):
    """
    Saves the top N rows of the dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): The dataframe containing the final results.
        filename (str): The name of the CSV file to save.
        num_rows (int): The number of top rows to save.
    """
    if not os.path.exists('output'):
        os.makedirs('output')
    
    output_df = df.head(num_rows)
    output_path = os.path.join('output', filename)
    
    output_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved top {num_rows} linebackers to '{output_path}'")

if __name__ == '__main__':
    # Ensure numpy settings for better display
    np.set_printoptions(suppress=True)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    try:
        # Phase 1: Data Preparation
        games, plays, players, pff_scouting = load_data(DATA_PATH)
        pbp_df, lb_pff = prepare_data(games, plays, players, pff_scouting, LB_POSITIONS)
        
        # Phase 2: Feature Engineering and WP Modeling
        pbp_df = feature_engineer(pbp_df)
        pbp_df = train_and_predict_wp(pbp_df)
        
        # Phase 3: WPA Calculation and Attribution
        final_lb_wpa = calculate_lb_wpa(pbp_df, lb_pff)

        print("\n" + "="*50)
        print("TOP 10 LINEBACKERS BY TOTAL WPA CONTRIBUTION")
        print("Total WPA is measured in probability of winning.")
        print("="*50)
        print(final_lb_wpa.head(10))
        
        # --- NEW CALL: Save the results to a CSV file
        save_results_to_csv(final_lb_wpa)

        print("\nAnalysis Complete.")

    except FileNotFoundError as e:
        print(f"\nERROR: One or more data files not found. Check your DATA_PATH configuration: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")
