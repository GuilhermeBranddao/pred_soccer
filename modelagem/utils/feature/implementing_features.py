import pandas as pd
import os
from modelagem.utils.logs import logger
from pathlib import Path

# from modelagem.feature_eng.match_analysis import get_storage_ranks
from modelagem.utils.feature.encode import (
    encode_categorical_features, 
)


from modelagem.feature_eng.strategy.basic import strategy_basic

FT_DIR = Path("database", "features")

def drop_unwanted_features(df: pd.DataFrame, drop_columns:list=None) -> pd.DataFrame:
    
    drop_columns = drop_columns or []

    return df.drop(columns=[col for col in drop_columns if col in df.columns])

def prep_data_to_save(df: pd.DataFrame, path_encoder: str, drop_columns:list[str]) -> pd.DataFrame:


    # Se ok deve salvar o que for necessario para realizar predições
    # encodes, features, ....

    # Codifica os times
    df = encode_categorical_features(df, path_save_encoder=path_encoder)
    df.fillna(0, inplace=True)
    # Drop de colunas descenessarias

    df = drop_unwanted_features(df, drop_columns=drop_columns)

    print(df.columns)

    logger.debug("Salvando o DataFrame resultante.")
    os.makedirs(FT_DIR, exist_ok=True)
    output_path = os.path.join(FT_DIR, 'ft_df.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Feature DataFrame salvo em {output_path}")
    return True, df



def create_feature_first_strategy(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    success, df = 'base_pre_processing'(df)

    if not success:
        return False, pd.DataFrame()

    # Se ok deve salvar o que for necessario para realizar predições
    # encodes, features, ....

    # Codifica os times
    df = encode_categorical_features(df, path_save_encoder=path_encoder)
    df.fillna(0, inplace=True)
    # Drop de colunas descenessarias

    drop_columns = ['match_name', 'datetime', 'home_team', 'away_team', 
  'home_score', 'away_score', 'h_match_points', 'a_match_points',
  'id', 'country', 'league', 'season', 'result', 'psch', 'pscd', 'psca',
       'maxch', 'maxcd', 'maxca', 'avgch', 'avgcd', 'avgca', 'bfech', 'bfecd',
       'hash', 'last_updated']
    df = drop_unwanted_features(df, drop_columns=drop_columns)

    logger.debug("Salvando o DataFrame resultante.")
    os.makedirs(FT_DIR, exist_ok=True)
    output_path = os.path.join(FT_DIR, 'ft_df.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Feature DataFrame salvo em {output_path}")
    return True, df

def strategy_basic(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    success, df = 'base_pre_processing'(df)
    if not success:
        return False, pd.DataFrame()

    df = encode_categorical_features(df, path_encoder)
    df.fillna(0, inplace=True)
    df = drop_unwanted_features(df)
    return True, df


def strategy_with_goal_stats(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    success, df = 'base_pre_processing'(df)
    if not success:
        return False, pd.DataFrame()

    df = encode_categorical_features(df, path_encoder)
    # df = add_goal_averages(df)  # Exemplo fictício
    df.fillna(0, inplace=True)
    df = drop_unwanted_features(df)
    return True, df


def strategy_experimental(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    success, df = 'base_pre_processing'(df)
    if not success:
        return False, pd.DataFrame()

    df = encode_categorical_features(df, path_encoder)
    # df = add_goal_averages(df)
    # df = add_win_rates(df)
    # df = calculate_ranking(df)
    df.fillna(0, inplace=True)
    df = drop_unwanted_features(df)
    return True, df

