import pandas as pd
import os
from modelagem.utils.logs import logger
from pathlib import Path

# from modelagem.feature_eng.match_analysis import get_storage_ranks
from modelagem.utils.feature.create_features import (
    get_recent_performance,
    add_home_away_stats,
    add_head_to_head_features,
    add_context_features,
    update_team_ranking,
    check_and_create_features,
    add_diff_features,
    calc_momentum_features,
    add_season_stats,
    add_temporal_features,
    encode_categorical_features,
    base_pre_processing
)

FT_DIR = Path("database", "features")

def selected_features_to_train(df:pd.DataFrame)->pd.DataFrame:
    drop_columns_catecorical = ["id","country","league","home_team","away_team",
    "result","psch","pscd","psca","maxch","maxcd","maxca","avgch","avgcd","avgca","bfech","bfecd","datetime",
    "hash","last_updated", "match_day_of_week", "season_phase"]

    drop_columns_numerical = ["home_score", "away_score"]

    drop_columns = drop_columns_catecorical
    drop_columns += drop_columns_numerical
    
    return df.drop(columns=drop_columns)


def create_first_strategy(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    success, df = base_pre_processing(df, path_encoder)


    logger.debug("Salvando o DataFrame resultante.")
    os.makedirs(FT_DIR, exist_ok=True)
    output_path = os.path.join(FT_DIR, 'ft_df.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Feature DataFrame salvo em {output_path}")
    return True, df

def create_feature(df: pd.DataFrame, path_encoder: str) -> pd.DataFrame:
    """
    Função principal para calcular as features de desempenho dos times.
    
    :param df: DataFrame contendo os dados das partidas.
    :param path_encoder: Caminho do arquivo onde o mapeamento será salvo.
    :return: DataFrame com as novas features adicionadas.
    """
    logger.debug("Inplementando features")
    
    # Calcula o desempenho recente
    df = get_recent_performance(df)

    # Adiciona estatísticas de casa e fora
    df = add_home_away_stats(df)

    # Adiciona estatísticas de confronto direto
    df = add_head_to_head_features(df)

    # Adiciona features contextuais
    df = add_context_features(df)

    # Atualiza o ranking dos times
    df = update_team_ranking(df)

    # Verifica e cria as features necessárias
    check_and_create_features(df)

    # Adiciona features de diferença
    df = add_diff_features(df)

    # Calcula as features de momentum
    df = calc_momentum_features(df, is_home=False, team_col='home_team')
    df = calc_momentum_features(df, is_home=False, team_col='away_team')

    # Adiciona estatísticas da temporada
    df = add_season_stats(df)
    # Adiciona features temporais
    df = add_temporal_features(df)

    # Codifica os times
    df = encode_categorical_features(df, path_encoder)

    df.fillna(0, inplace=True)

    # Drop de colunas descenessarias 
    df = selected_features_to_train(df)


    logger.debug("Salvando o DataFrame resultante.")
    os.makedirs(FT_DIR, exist_ok=True)
    output_path = os.path.join(FT_DIR, 'ft_df.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Feature DataFrame salvo em {output_path}")
    return True, df