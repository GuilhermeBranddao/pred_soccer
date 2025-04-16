from modelagem.utils.feature.encode import (
    encode_categorical_features, 
)
from modelagem.utils.feature.tool_kit import prep_data_to_save, drop_unwanted_features
import pandas as pd

def strategy_basic_2(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
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

