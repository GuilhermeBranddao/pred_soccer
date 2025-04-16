import pandas as pd
import numpy as np
from modelagem.utils.logs import logger
from modelagem.utils.feature.tool_kit import prep_data_to_save
from pathlib import Path

def main(df:pd.DataFrame, path_encoder:Path, columns_request: list[str] = None) -> tuple[bool, str | pd.DataFrame]:
    """
    Função para pré-processar os dados de futebol.
    Ela realiza as seguintes etapas:
    1. Carrega os dados de um DataFrame.
    2. Cria colunas iniciais, como 'match_name' e 'datetime'.
    3. Seleciona colunas importantes para análise.
    4. Converte colunas para o tipo inteiro.
    5. Realiza o encoding dos times.
    6. Calcula os pontos e resultados das partidas.
    7. Realiza feature engineering para criar novas colunas.
    8. Salva o DataFrame resultante em um arquivo CSV.
    9. Retorna True se o pré-processamento for bem-sucedido, caso contrário, retorna False.
    
    :param df: DataFrame contendo os dados das partidas.
    :return: True se o pré-processamento for bem-sucedido, False caso contrário.
    """
    df = df.copy()  # Evita o warning

    logger.info("Iniciando pré-processamento dos dados...")
    
    if df is None or df.empty:
        logger.error("Nenhum dado foi carregado. Encerrando pré-processamento.")
        return False, 'Nenhum dado foi carregado'

    try:
        # Criando colunas iniciais
        df['match_name'] = df['home_team'] + ' - ' + df['away_team']
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Convertendo colunas para inteiro
        to_int = ['season', 'home_score', 'away_score']
        df[to_int] = df[to_int].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)


        # Calculando pontos e resultado das partidas
        logger.debug("Iniciando o cálculo dos pontos e resultados das partidas.")
        df = calculate_match_points(df)

        # Feature Engineering
        logger.debug("Iniciando o feature engineering.")
        df_storage_ranks = get_storage_ranks(df)

        logger.debug("Iniciando o feature engineering do time da casa.")
        ht_cols = [f'ht{col}' for col in get_feature_columns()]
        at_cols = [f'at{col}' for col in get_feature_columns()]

        df[ht_cols] = df.apply(lambda x: create_main_cols(x, x.home_team, df, df_storage_ranks), axis=1, result_type='expand')
        df[at_cols] = df.apply(lambda x: create_main_cols(x, x.away_team, df, df_storage_ranks), axis=1, result_type='expand')

        # Removendo colunas desnecessárias

        df.fillna(-33, inplace=True)  # Preenchendo valores ausentes

        columns_request = columns_request or []

        err, df = prep_data_to_save(df=df, 
                                    path_encoder=path_encoder, 
                                    columns_request=columns_request)

        #logger.info(f"Implementação de feature finalizado")
        return err, df

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}", exc_info=True)
        return False, 'Erro no pré-processamento'


def get_storage_ranks(df):
    """
    OBJETIVO:
        - A função get_rank processa os dados de uma temporada anterior, agrupa as informações relevantes, calcula o saldo de gols, e classifica os times com base em pontos, saldo de gols e número de gols. 
        O ranking final do time especificado é então retornado.

        - É parecido com a tabela do brasileirão com o rank dos times
    PARAMETERS:
        - x: Um registro ou linha de dados contendo a temporada atual e outras informações.
        - team: O time para o qual o ranking será calculado.
        - delta_year: O número de anos de diferença entre a temporada de referência e a temporada para a qual o ranking será calculado.
    RETURN:
        - team_rank: A posição (ranking) do time na temporada ajustada por delta_year.
    """
    list_season_year = df['season'].unique()

    df_storage_ranks = pd.DataFrame()

    for season_year in list_season_year:
        df_full_season = df[(df['season'] == (season_year))].drop(columns='datetime')

        # Agrupamento de Dados dos Jogos em Casa
        # - Agrupa os dados pelo time da casa (home_team), somando os pontos e gols em casa (home_score) e os gols sofridos em casa (away_score). As colunas são renomeadas para facilitar o entendimento.
        df_full_home = df_full_season.groupby(['home_team']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()
        df_full_home.columns = ['team', 'points', 'goals', 'goals_sf']

        # Agrupamento de Dados dos Jogos Fora de Casa
        # - Similar ao passo anterior, mas agora para os jogos fora de casa (away_team). As colunas são novamente renomeadas.
        df_full_away = df_full_season.groupby(['away_team']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()
        df_full_away.columns = ['team', 'points', 'goals', 'goals_sf']

        # Concatena os DataFrames e soma as colunas numéricas
        df_concat = pd.concat([df_full_home, df_full_away], ignore_index=True)

        # Agrupa pelo nome do time e soma as colunas numéricas
        df_rank = df_concat.groupby('team', as_index=False).sum()

        # Cálculo do Saldo de Gols (subtraindo os gols sofridos dos gols marcados)
        df_rank['goals_df'] = df_rank.goals - df_rank.goals_sf
        # Reagrupamento e Soma dos Resultados
        # - Reagrupa os dados pelo time, somando os pontos, gols, e saldo de gols para obter um resumo final por time.
        df_rank = df_rank.groupby(['team']).sum().reset_index()
        # Ordenação e Ranking
        # - Ordena os times por pontos, saldo de gols e número total de gols (nesta ordem de prioridade). Em seguida, cria uma coluna rank que atribui a posição no ranking.
        df_rank = df_rank.sort_values(by = ['points', 'goals_df', 'goals'], ascending = False)
        df_rank['rank'] = df_rank.points.rank(method = 'first', ascending = False).astype(int)

        df_rank['season'] = season_year

        df_storage_ranks = pd.concat([df_storage_ranks, df_rank])

    return df_storage_ranks


def get_rank(x:pd.Series, team:str, delta_year:int, df_storage_ranks:pd.DataFrame):
    # Filtrando por temporada
    df_rank = df_storage_ranks[(df_storage_ranks.season == (int(x.season) - delta_year))]

    team_rank = df_rank[df_rank.team == team].min()['rank']

    return team_rank

def get_match_stats(x:pd.Series, team:str, df:pd.DataFrame):
    """
    OBJETIVO:
    - A função get_match_stats calcula várias estatísticas de desempenho de um time ao longo de uma temporada até uma data específica.
    - A função tem como objetivo calcular estatísticas acumuladas e médias ponderadas para um time em uma temporada até uma data específica. 
        - As estatísticas incluem pontos totais, gols, saldo de gols, vitórias, empates, derrotas e sequências de vitórias, empates ou derrotas.
    
    PARAMETERS:
        - x: Uma linha específica de um DataFrame, que contém informações sobre um jogo, incluindo a data e a temporada.
        - team: O time para o qual as estatísticas serão calculadas.
    RETURN:
        - A função retorna uma série de estatísticas relacionadas ao desempenho do time, incluindo pontos, gols, saldo de gols, vitórias, empates, derrotas e médias ponderadas em uma janela de tempo recente.
    """
    # Filtragem de Jogos em Casa e Fora de Casa
    home_df = df[(df.home_team == team) & (df.datetime < x.datetime) & (df.season == x.season)]
    away_df = df[(df.away_team == team) & (df.datetime < x.datetime) & (df.season == x.season)]

    # Agrupamento e Cálculo de Estatísticas para Jogos em Casa
    home_table = home_df.groupby(['datetime']).sum()[['h_match_points', 'home_score', 'away_score']].reset_index()
    home_table.columns = ['datetime', 'points', 'goals', 'goals_sf']
    home_table['goals_df'] = home_table.goals - home_table.goals_sf
    home_table['host'] = 'home'

    # Agrupamento e Cálculo de Estatísticas para Jogos em Fora de Casa
    away_table = away_df.groupby(['datetime']).sum()[['a_match_points', 'away_score', 'home_score']].reset_index()
    away_table.columns = ['datetime', 'points', 'goals', 'goals_sf']
    away_table['goals_df'] = away_table.goals - away_table.goals_sf
    away_table['host'] = 'away'

    # Combinação e Ordenação dos Dados
    # - Combina os dados dos jogos em casa e fora em um único DataFrame e ordena os jogos pela data.
    full_table = pd.concat([home_table, away_table], ignore_index = True)
    full_table = full_table.sort_values('datetime', ascending = True)


    # Cálculo de Sequências de Vitórias, Empates e Derrotas
    # - Identifica onde as sequências de vitórias, empates ou derrotas começam e atribui um streak_id para cada sequência. Depois, calcula o tamanho de cada sequência (streak_counter).
    # FIXME: Pra mim esse trecho não faz sentiudo, pois ele calcula tudo, vitoras, empates ou derrotas, pra mim faz mais sentido ou separar essas informações em outras tabelas ou manter apenas a sequencia de vitoria
    full_table['start_of_streak'] = full_table.points.ne(full_table.points.shift())
    full_table['streak_id'] = full_table['start_of_streak'].cumsum()
    full_table['streak_counter'] = full_table.groupby('streak_id').cumcount() + 1

    # Cálculo das Médias Ponderadas Exponenciais
    # - Calcula médias ponderadas exponenciais para pontos, gols marcados e gols sofridos usando um span de 3, o que dá mais peso aos jogos mais recentes.
    # - A função ewm(span=3) aplica a média ponderada com um decaimento exponencial. Para cada jogo, a média ponderada exponencial é calculada com base nos jogos anteriores, mas dá mais peso ao mais recente.
    full_table['w_avg_points'] = full_table.points.ewm(span=3, adjust=False).mean()
    full_table['w_avg_goals'] = full_table.goals.ewm(span=3, adjust=False).mean()
    full_table['w_avg_goals_sf'] = full_table.goals_sf.ewm(span=3, adjust=False).mean()

    # Identificação das Sequências Atuais
    # - Filtra o DataFrame para manter apenas as informações sobre a data mais recente (último jogo).
    streak_table = full_table[full_table.datetime == full_table.datetime.max()]

    # Determinação da Natureza da Sequência (Vitória, Empate ou Derrota)
    # - Com base nos pontos ganhos no último jogo, determina se o time está em uma sequência de vitórias, empates ou derrotas e conta o tamanho dessa sequência.
    if streak_table.points.min() == 3:
        win_streak = streak_table.streak_counter.sum()
        loss_streak = 0
        draw_streak = 0
    elif streak_table.points.min() == 0:
        win_streak = 0
        loss_streak = streak_table.streak_counter.sum()
        draw_streak = 0
    else:
        win_streak = 0
        loss_streak = 0
        draw_streak = streak_table.streak_counter.sum()
    
    # Cálculo de Estatísticas Totais
    # - Calcula as estatísticas totais de pontos, gols, saldo de gols, vitórias, empates e derrotas para jogos em casa e fora.
    home_points = home_table.points.sum()
    home_goals = home_table.goals.sum()
    home_goals_sf = home_table.goals_sf.sum()
    home_wins = len(home_table[home_table.points == 3])
    home_draws = len(home_table[home_table.points == 1])
    home_losses = len(home_table[home_table.points == 0])

    away_points = away_table.points.sum()
    away_goals = away_table.goals.sum()
    away_goals_sf = away_table.goals_sf.sum()
    away_wins = len(away_table[away_table.points == 3])
    away_draws = len(away_table[away_table.points == 1])
    away_losses = len(away_table[away_table.points == 0])

    total_points = home_points + away_points
    total_goals = home_goals + away_goals
    total_goals_sf = home_goals_sf + away_goals_sf
    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses
    
    # Cálculo de Estatísticas Recentes (Últimos 3 Jogos)
    # - Filtra os últimos 3 jogos e calcula as estatísticas médias para pontos, gols, e saldo de gols, tanto no agregado quanto ponderado exponencialmente.
    full_table_delta = full_table[full_table.datetime.isin(full_table.datetime[-3:])]

    home_l_points = full_table_delta[full_table_delta.host == 'home'].points.sum()
    away_l_points = full_table_delta[full_table_delta.host == 'away'].points.sum()

    #total metric in given delta averaged
    total_l_points = (home_l_points + away_l_points)/3
    total_l_goals = (home_goals + away_goals)/3
    total_l_goals_sf = (home_goals_sf + away_goals)/3

    total_l_w_avg_points = full_table[full_table.datetime.isin(full_table.datetime[-1:])].w_avg_points.sum()
    total_l_w_avg_goals = full_table[full_table.datetime.isin(full_table.datetime[-1:])].w_avg_goals.sum()
    total_l_w_avg_goals_sf = full_table[full_table.datetime.isin(full_table.datetime[-1:])].w_avg_goals_sf.sum()

    # Retorno das Estatísticas
    return total_points, total_l_points, total_l_w_avg_points, total_goals, total_l_goals, total_l_w_avg_goals, total_goals_sf, total_l_goals_sf, total_l_w_avg_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak

def get_match_stats_optimized(x: pd.Series, team: str, df: pd.DataFrame):
    # Filtragem inicial
    matches = df[(df.datetime < x.datetime) & (df.season == x.season)].copy()
    
    # Retorne valores padrão caso jogos não sejam encontrados
    if matches.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  

    # Separar jogos em casa e fora
    matches['is_home'] = matches.home_team == team
    matches['points'] = matches.apply(lambda row: row.h_match_points if row.is_home else row.a_match_points, axis=1)
    matches['goals'] = matches.apply(lambda row: row.home_score if row.is_home else row.away_score, axis=1)
    matches['goals_sf'] = matches.apply(lambda row: row.away_score if row.is_home else row.home_score, axis=1)
    matches['goals_df'] = matches['goals'] - matches['goals_sf']
    
    # Ordenar por data
    matches = matches.sort_values('datetime')

    # Estatísticas totais
    total_points = matches.points.sum()
    total_goals = matches.goals.sum()
    total_goals_sf = matches.goals_sf.sum()
    total_wins = (matches.points == 3).sum()
    total_draws = (matches.points == 1).sum()
    total_losses = (matches.points == 0).sum()

    
    # Últimos 3 jogos
    recent_matches = matches.tail(3)
    total_l_points = recent_matches.points.mean()
    total_l_goals = recent_matches.goals.mean()
    total_l_goals_sf = recent_matches.goals_sf.mean()

    # Médias ponderadas exponenciais
    w_avg_points = matches.points.ewm(span=3, adjust=False).mean().iloc[-1]
    w_avg_goals = matches.goals.ewm(span=3, adjust=False).mean().iloc[-1]
    w_avg_goals_sf = matches.goals_sf.ewm(span=3, adjust=False).mean().iloc[-1]

    # Sequências
    streak_id = (matches.points != matches.points.shift()).cumsum()
    streak_data = matches.groupby(streak_id).points.agg(['first', 'size'])
    current_streak = streak_data.iloc[-1]
    win_streak = current_streak['size'] if current_streak['first'] == 3 else 0
    draw_streak = current_streak['size'] if current_streak['first'] == 1 else 0
    loss_streak = current_streak['size'] if current_streak['first'] == 0 else 0

    # Retorno
    return (
        total_points, total_l_points, w_avg_points, total_goals, 
        total_l_goals, w_avg_goals, total_goals_sf, total_l_goals_sf, 
        w_avg_goals_sf, total_wins, total_draws, total_losses, 
        win_streak, loss_streak, draw_streak
    )

def get_days_ls_match(x, team, df):
    """
    OBJETIVO:

    PARAMETERS:

    RETURN:
    """
    #filtering last game of the team and getting datetime
    last_date = df[(df.datetime < x.datetime) & (df.season == x.season) & (df.match_name.str.contains(team))].datetime.max()

    days = (x.datetime - last_date)/np.timedelta64(1,'D')

    return days

def get_ls_winner(x, df):
    """
    OBJETIVO:

    PARAMETERS:

    RETURN:
    """
    temp_df = df[(df.datetime < x.datetime) & (df.match_name.str.contains(x.home_team)) & (df.match_name.str.contains(x.away_team))]
    temp_df = temp_df[temp_df.datetime == temp_df.datetime.max()]
    
    #checking if there was a previous match
    if len(temp_df) == 0:
        result = None
    elif temp_df.winner.all() == 'DRAW':
        result = 'DRAW'
    elif temp_df.home_team.all() == x.home_team:
        result = temp_df.winner.all()
    else:
        if temp_df.winner.all() == 'HOME_TEAM':
            result = 'HOME_TEAM'
        else:
            result = 'AWAY_TEAM'
    
    return result

def create_main_cols(x:pd.Series, team:str, df:pd.DataFrame, df_storage_ranks:pd.DataFrame):
    """
    OBJETIVO:

    PARAMETERS:
        - x: 1 linha do dataframe
        - team: time
        
    RETURN:
    """
    
    #get current and last delta (years) rank
    team_rank = get_rank(x, team, 0, df_storage_ranks=df_storage_ranks)
    ls_team_rank = get_rank(x, team, 1, df_storage_ranks=df_storage_ranks)

    #get main match stats

    (total_points, 
    total_l_points, 
    total_l_w_avg_points, 
    total_goals, 
    total_l_goals, 
    total_l_w_avg_goals, 
    total_goals_sf, 
    total_l_goals_sf, 
    total_l_w_avg_goals_sf, 
    total_wins, 
    total_draws, 
    total_losses, 
    win_streak, 
    loss_streak,
    draw_streak) = get_match_stats_optimized(x, team, df)

    #get days since last match
    days = get_days_ls_match(x, team, df)    

    return team_rank, ls_team_rank, days, total_points, total_l_points, total_l_w_avg_points, total_goals, total_l_goals, total_l_w_avg_goals, total_goals_sf, total_l_goals_sf, total_l_w_avg_goals_sf, total_wins, total_draws, total_losses, win_streak, loss_streak, draw_streak



def calculate_match_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para calcular o vencedor de uma partida e os pontos correspondentes.
    Ela adiciona colunas ao DataFrame original para indicar o vencedor e os pontos ganhos por cada time.

    :param df: DataFrame contendo os dados das partidas.
    :return: DataFrame com colunas adicionais para o vencedor e os pontos.
    """
    
    df = df.copy()

    dict_result = {'DRAW': 0, 'AWAY_WIN': 1, 'HOME_WIN': 2}
    
    # Garantir que df seja uma cópia independente
    df = df.copy()

    df.loc[:, 'winner'] = np.select(
        [df['home_score'] > df['away_score'], df['home_score'] < df['away_score']],
        [dict_result["HOME_WIN"], dict_result["AWAY_WIN"]],
        default=dict_result['DRAW']
    )

    df.loc[:, 'h_match_points'] = np.select(
        [df['winner'] == dict_result['HOME_WIN'], df['winner'] == dict_result['DRAW']],
        [3, 1],
        default=0
    )

    df.loc[:, 'a_match_points'] = np.select(
        [df['winner'] == dict_result['AWAY_WIN'], df['winner'] == dict_result['DRAW']],
        [3, 1],
        default=0
    )

    return df

def get_feature_columns():
    """Retorna a lista de colunas usadas na engenharia de features"""
    return ['_rank', '_ls_rank', '_days_ls_match', '_points', '_l_points', 
            '_l_wavg_points', '_goals', '_l_goals', '_l_wavg_goals', '_goals_sf', 
            '_l_goals_sf', '_l_wavg_goals_sf', '_wins', '_draws', '_losses', 
            '_win_streak', '_loss_streak', '_draw_streak']