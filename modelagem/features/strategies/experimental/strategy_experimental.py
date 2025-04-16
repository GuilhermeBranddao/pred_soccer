from modelagem.utils.logs import logger
import pandas as pd
from modelagem.utils.feature.tool_kit import prep_data_to_save
from modelagem.utils.feature.encode import (
    encode_categorical_features, 
)
from pathlib import Path
# from modelagem.feature_eng.match_analysis import get_storage_ranks
from modelagem.features.strategies.experimental.create_features import (
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
    # base_pre_processing
)

def main(df: pd.DataFrame, path_encoder: Path, columns_request: list[str] = None) -> tuple[bool, str | pd.DataFrame]:
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
    df = encode_categorical_features(df, path_save_encoder=path_encoder)
    df.fillna(0, inplace=True)

    columns_request = columns_request or []

    err, df = prep_data_to_save(df=df, 
                                    path_encoder=path_encoder, 
                                    columns_request=columns_request)

    return err, df