# main_trainer.py

import os
import pandas as pd
from pathlib import Path

# from database.create_connection import create_connection
from modelagem.utils.logs import logger
from modelagem.utils.get_data import get_soccer_data
from modelagem.train import model_trainer

# from modelagem.utils.feature.encode import base_pre_processing
from modelagem.features.strategies.basic import strategy_basic
from modelagem.features.strategies.experimental import strategy_experimental

# Definindo diretórios base
# DATA_DIR = os.path.join('feature_eng', 'data')
# MODEL_DIR = os.path.join('database', 'models')
# LOG_DIR = os.path.join('database', 'logs')
# # Diretórios
# FT_DIR = Path("database", "features")
from modelagem.settings.config import Settings
config = Settings()

FEATURE_STRATEGIES:list[dict] = [
    {"feature_name":"first", 
     "func_feature":strategy_experimental.main, 
     "coluns_request":["season", 'home_team_last_5_win_rate', 'away_team_last_5_win_rate',
       'home_team_goal_avg_last_5', 'away_team_goal_avg_last_5',
       'home_team_goals_conceded_last_5', 'away_team_goals_conceded_last_5',
       'home_team_form', 'away_team_form', 'home_team_home_win_rate',
       'away_team_away_win_rate', 'home_team_home_goal_avg',
       'away_team_away_goal_avg', 'head_to_head_win_home_team',
       'head_to_head_draws', 
       'head_to_head_goal_diff', 'is_weekend', 'month',
       'days_since_last_match_home', 'match_importance', 'home_team_ranking',
       'away_team_ranking', 'goal_avg_diff', 'win_rate_diff', 'ranking_diff',
       'away_team_streak', 'away_team_last_match_result',
       'away_team_points_last_3', 'away_team_score_diff_last_5',
       'home_team_total_goals_scored_season',
       'home_team_total_goals_conceded_season',
       'home_team_matches_played_season', 'home_team_goal_diff_season',
       'away_team_total_goals_scored_season',
       'away_team_total_goals_conceded_season', 'home_team_encoded',
       'away_team_encoded', 'match_day_of_week_encoded',
       'season_phase_encoded', "result_encoded"]
     },
    {"feature_name":"basic", 
    "func_feature":strategy_basic.main, 
    "coluns_request":['season', 'ht_rank', 'ht_ls_rank', 'ht_days_ls_match', 'ht_points', 'ht_l_points',
       'ht_l_wavg_points', 'ht_goals', 'ht_l_goals', 'ht_l_wavg_goals',
       'ht_goals_sf', 'ht_l_goals_sf', 'ht_l_wavg_goals_sf', 'ht_wins',
       'ht_draws', 'ht_losses', 'ht_win_streak', 'ht_loss_streak',
       'ht_draw_streak', 'at_rank', 'at_ls_rank', 'at_days_ls_match',
       'at_points', 'at_l_points', 'at_l_wavg_points', 'at_goals',
       'at_l_goals', 'at_l_wavg_goals', 'at_goals_sf', 'at_l_goals_sf',
       'at_l_wavg_goals_sf', 'at_wins', 'at_draws', 'at_losses',
       'at_win_streak', 'at_loss_streak', 'at_draw_streak',
       'home_team_encoded', 'away_team_encoded', 'result_encoded']}
]

def run_feature_engineering(df, strategy_idx=1):
    strategy = FEATURE_STRATEGIES[strategy_idx]
    func_feature = strategy["func_feature"]
    columns_request = strategy["coluns_request"]
    return func_feature(df, 
                        path_encoder=config.MAPPING_DIR,
                        columns_request=columns_request)

def load_features():
    return pd.read_csv(os.path.join(config.FT_DIR, 'ft_df.csv'))

def train_model(df):
    model_trainer.main(df)


if __name__ == "__main__":
    df = get_soccer_data()
    success, df_features = run_feature_engineering(df)

    if success:
        logger.info("Pré-processamento concluído com sucesso!")
        # df_features = load_features()
        logger.info("Iniciando treinamento")
        train_model(df_features)
    else:
        logger.error("Erro no pré-processamento.")

