import pandas as pd
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
)


def main(df: pd.DataFrame, path_team_mapping: str) -> pd.DataFrame:
    """
    Função principal para calcular as features de desempenho dos times.
    
    :param df: DataFrame contendo os dados das partidas.
    :param path_team_mapping: Caminho do arquivo onde o mapeamento será salvo.
    :return: DataFrame com as novas features adicionadas.
    """

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
    df = encode_categorical_features(df, path_team_mapping)
   
    return df
