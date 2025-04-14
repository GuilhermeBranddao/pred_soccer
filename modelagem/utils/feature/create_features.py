import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

def get_recent_performance(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Essas features são muito úteis e não tão difíceis de implementar. Vamos por partes. 
    Abaixo está uma função para calcular essas features de desempenho recente para todos os times em todo o dataset, baseada no histórico até aquele jogo.
    """
    df = df.copy()
    df = df.sort_values("datetime")

    teams = set(df['home_team']).union(df['away_team'])

    stats = {
        team: {
            'results': [],
            'goals_for': [],
            'goals_against': []
        } for team in teams
    }

    # Listas para armazenar as novas features
    features = {
        'home_team_last_5_win_rate': [],
        'away_team_last_5_win_rate': [],
        'home_team_goal_avg_last_5': [],
        'away_team_goal_avg_last_5': [],
        'home_team_goals_conceded_last_5': [],
        'away_team_goals_conceded_last_5': [],
        'home_team_form': [],
        'away_team_form': []
    }

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        # Calcula features para o time da casa
        home_stats = stats[home]
        recent_home_results = home_stats['results'][-n:]
        recent_home_goals = home_stats['goals_for'][-n:]
        recent_home_conceded = home_stats['goals_against'][-n:]

        win_rate_home = recent_home_results.count('W') / n if recent_home_results else 0.0
        avg_goals_home = np.mean(recent_home_goals) if recent_home_goals else 0.0
        avg_conceded_home = np.mean(recent_home_conceded) if recent_home_conceded else 0.0
        form_home = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in recent_home_results])

        # Calcula features para o time visitante
        away_stats = stats[away]
        recent_away_results = away_stats['results'][-n:]
        recent_away_goals = away_stats['goals_for'][-n:]
        recent_away_conceded = away_stats['goals_against'][-n:]

        win_rate_away = recent_away_results.count('W') / n if recent_away_results else 0.0
        avg_goals_away = np.mean(recent_away_goals) if recent_away_goals else 0.0
        avg_conceded_away = np.mean(recent_away_conceded) if recent_away_conceded else 0.0
        form_away = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in recent_away_results])

        # Armazena as features
        features['home_team_last_5_win_rate'].append(win_rate_home)
        features['away_team_last_5_win_rate'].append(win_rate_away)
        features['home_team_goal_avg_last_5'].append(avg_goals_home)
        features['away_team_goal_avg_last_5'].append(avg_goals_away)
        features['home_team_goals_conceded_last_5'].append(avg_conceded_home)
        features['away_team_goals_conceded_last_5'].append(avg_conceded_away)
        features['home_team_form'].append(form_home)
        features['away_team_form'].append(form_away)

        # Atualiza os stats depois do jogo
        if home_score > away_score:
            home_result, away_result = 'W', 'L'
        elif home_score < away_score:
            home_result, away_result = 'L', 'W'
        else:
            home_result = away_result = 'D'

        home_stats['results'].append(home_result)
        home_stats['goals_for'].append(home_score)
        home_stats['goals_against'].append(away_score)

        away_stats['results'].append(away_result)
        away_stats['goals_for'].append(away_score)
        away_stats['goals_against'].append(home_score)

    # Adiciona ao DataFrame
    for col, values in features.items():
        df[col] = values

    return df

def add_home_away_stats(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Essas features são excelentes pra capturar o efeito do mando de campo, que realmente faz diferença em muitos jogos. Vamos implementar esse conjunto agora.
    """
    df = df.copy()
    df = df.sort_values("datetime")

    teams = set(df['home_team']).union(df['away_team'])

    home_stats = {team: {'results': [], 'goals': []} for team in teams}
    away_stats = {team: {'results': [], 'goals': []} for team in teams}

    features = {
        'home_team_home_win_rate': [],
        'away_team_away_win_rate': [],
        'home_team_home_goal_avg': [],
        'away_team_away_goal_avg': []
    }

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        # Últimos N jogos em casa
        home_r = home_stats[home]['results'][-n:]
        home_g = home_stats[home]['goals'][-n:]
        home_win_rate = home_r.count('W') / n if home_r else 0.0
        home_goal_avg = np.mean(home_g) if home_g else 0.0

        # Últimos N jogos fora
        away_r = away_stats[away]['results'][-n:]
        away_g = away_stats[away]['goals'][-n:]
        away_win_rate = away_r.count('W') / n if away_r else 0.0
        away_goal_avg = np.mean(away_g) if away_g else 0.0

        features['home_team_home_win_rate'].append(home_win_rate)
        features['away_team_away_win_rate'].append(away_win_rate)
        features['home_team_home_goal_avg'].append(home_goal_avg)
        features['away_team_away_goal_avg'].append(away_goal_avg)

        # Atualiza os históricos
        if home_score > away_score:
            home_result, away_result = 'W', 'L'
        elif home_score < away_score:
            home_result, away_result = 'L', 'W'
        else:
            home_result = away_result = 'D'

        home_stats[home]['results'].append(home_result)
        home_stats[home]['goals'].append(home_score)

        away_stats[away]['results'].append(away_result)
        away_stats[away]['goals'].append(away_score)

    for col, values in features.items():
        df[col] = values

    return df


def add_head_to_head_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    As features de confronto direto (head-to-head) são ótimas pra capturar rivalidades ou padrões históricos entre dois times específicos. 
    3 features de head-to-head (confrontos diretos) com base no seu histórico de jogos.
    Vamos montar isso com os últimos X jogos entre eles (você pode ajustar o valor de X).
    """
    df = df.copy()
    df = df.sort_values("datetime")

    # Mapeia o histórico entre pares de times
    h2h_history = {}

    # Features para armazenar
    head_to_head_win_home_team = []
    head_to_head_draws = []
    head_to_head_goal_diff = []

    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        key = tuple(sorted([home, away]))  # ex: ('Flamengo RJ', 'Palmeiras')

        # Recupera o histórico entre os dois
        matches = h2h_history.get(key, [])[-n:]

        # Calcula vitórias do mandante (considerando se ele foi o time 1 ou 2 no histórico)
        win_count = 0
        draw_count = 0
        goal_diff_total = 0

        for m in matches:
            h, a, hs, as_ = m

            # Quem foi mandante naquele confronto?
            if h == home:
                if hs > as_:
                    win_count += 1
                elif hs == as_:
                    draw_count += 1
                goal_diff_total += (hs - as_)
            else:
                if as_ > hs:
                    win_count += 1
                elif hs == as_:
                    draw_count += 1
                goal_diff_total += (as_ - hs)

        # Adiciona as features calculadas
        head_to_head_win_home_team.append(win_count)
        head_to_head_draws.append(draw_count)
        head_to_head_goal_diff.append(goal_diff_total)

        # Atualiza o histórico
        if key not in h2h_history:
            h2h_history[key] = []

        h2h_history[key].append((home, away, home_score, away_score))

    # Atribui ao DataFrame
    df['head_to_head_win_home_team'] = head_to_head_win_home_team
    df['head_to_head_draws'] = head_to_head_draws
    df['head_to_head_goal_diff'] = head_to_head_goal_diff

    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Garantir que a coluna datetime esteja no formato certo
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    # Feature: Jogo é no final de semana?
    df['is_weekend'] = df['datetime'].dt.weekday >= 5  # Sábado = 5, Domingo = 6

    # Feature: Mês do jogo
    df['month'] = df['datetime'].dt.month

    # Dicionários para guardar a última data de cada time
    last_match_home = {}
    days_since_last_home = []

    for idx, row in df.iterrows():
        home_team = row['home_team']
        current_date = row['datetime']

        # Dias desde o último jogo do time da casa
        if home_team in last_match_home:
            delta = (current_date - last_match_home[home_team]).days
        else:
            delta = np.nan  # primeiro jogo do time, sem histórico

        days_since_last_home.append(delta)
        last_match_home[home_team] = current_date  # atualiza última data

    df['days_since_last_match_home'] = days_since_last_home

    # Feature opcional: match_importance (placeholder)
    # Aqui você pode criar uma regra simples como: jogos no fim da temporada são mais importantes
    # Exemplo simples: jogos após o mês 10 (outubro) valem mais
    df['match_importance'] = (df['month'] >= 10).astype(int)

    return df

def check_and_create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Verifica se as colunas necessárias existem
    required_columns = {
        'goal_avg': ['home_team_goal_avg_last_5', 'away_team_goal_avg_last_5'],
        'win_rate': ['home_team_last_5_win_rate', 'away_team_last_5_win_rate'],
        'ranking': ['home_team_ranking', 'away_team_ranking']
    }

    # Aviso sobre as colunas ausentes
    if not all(col in df.columns for col in required_columns['goal_avg']):
        print("⚠️ Colunas de goal_avg não encontradas!")
    if not all(col in df.columns for col in required_columns['win_rate']):
        print("⚠️ Colunas de win_rate não encontradas!")
    if not all(col in df.columns for col in required_columns['ranking']):
        print("ℹ️ Colunas de ranking não encontradas (pulando ranking_diff)!")

    # Agora, criamos as features apenas se as colunas forem encontradas
    if all(col in df.columns for col in required_columns['goal_avg']):
        # Exemplo de feature que usa a diferença de gols
        df['goal_avg_diff'] = df['home_team_goal_avg_last_5'] - df['away_team_goal_avg_last_5']
    
    if all(col in df.columns for col in required_columns['win_rate']):
        # Exemplo de feature que usa a diferença de taxa de vitória
        df['win_rate_diff'] = df['home_team_last_5_win_rate'] - df['away_team_last_5_win_rate']
    
    if all(col in df.columns for col in required_columns['ranking']):
        # Exemplo de feature que usa a diferença de ranking
        df['ranking_diff'] = df['home_team_ranking'] - df['away_team_ranking']

    return df


def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Observações:
    As diferenças são calculadas como home - away, então valores positivos significam vantagem para o mandante.

    O ranking_diff é feito como away - home porque rankings menores são melhores. Assim, se ranking_diff > 0, o time da casa está melhor ranqueado.

    :param df: DataFrame com as colunas: ['home_team_goal_avg', 'away_team_goal_avg', 'home_team_win_rate', 'away_team_win_rate', 'home_team_rank', 'away_team_rank']
    :return: DataFrame com as novas colunas de diferença
    """
    df = df.copy()

    # Diferença na média de gols marcados 

    if 'home_team_goal_avg_last_5' in df.columns and 'away_team_goal_avg_last_5' in df.columns:
        df['goal_avg_diff'] = df['home_team_goal_avg_last_5'] - df['away_team_goal_avg_last_5']
    else:
        print("⚠️ Colunas de goal_avg não encontradas!")

    # Diferença nas taxas de vitória
    if 'home_team_last_5_win_rate' in df.columns and 'away_team_last_5_win_rate' in df.columns:
        df['win_rate_diff'] = df['home_team_last_5_win_rate'] - df['away_team_last_5_win_rate']
    else:
        print("⚠️ Colunas de win_rate não encontradas!")

    # Diferença de ranking (quanto menor, melhor rank)
    if 'home_team_rank' in df.columns and 'away_team_rank' in df.columns:
        df['ranking_diff'] = df['away_team_rank'] - df['home_team_rank']
    else:
        print("ℹ️ Colunas de ranking não encontradas (pulando ranking_diff)")

    return df


def calc_momentum_features(df: pd.DataFrame, team_col: str, is_home: bool) -> pd.DataFrame:
    """
    Calcula features de momentum e forma recente.
    :param df: DataFrame ordenado cronologicamente com as colunas: [team_col, 'home_score', 'away_score', 'result']
    :param team_col: nome da coluna do time (home_team ou away_team)
    :param is_home: se True, calcula para o time da casa
    """
    df = df.copy()
    streaks = []
    last_result = []
    last_points_3 = []
    score_diff_5 = []

    team_results = {}

    for idx, row in df.iterrows():
        team = row[team_col]
        is_team_home = is_home
        opponent_score = row['away_score'] if is_team_home else row['home_score']
        team_score = row['home_score'] if is_team_home else row['away_score']

        # Determina resultado
        if team_score > opponent_score:
            res = 1
        elif team_score == opponent_score:
            res = 0
        else:
            res = -1

        # Atualiza histórico do time
        if team not in team_results:
            team_results[team] = []

        # Calcula streak
        prev_results = team_results[team][-10:]  # histórico recente
        streak = 0
        for r in reversed(prev_results):
            if r == 1:
                streak += 1
            else:
                break

        points_last_3 = sum([3 if r == 1 else 1 if r == 0 else 0 for r in team_results[team][-3:]])
        score_diffs = team_results[team][-5:]
        score_diff_sum = sum([r for r in score_diffs if isinstance(r, int)])  # saldo

        # Adiciona aos vetores
        streaks.append(streak)
        last_result.append(team_results[team][-1] if team_results[team] else 0)
        last_points_3.append(points_last_3)
        score_diff_5.append(score_diff_sum)

        # Salva resultado para próximas iterações
        team_results[team].append(res)
    
    prefix = "home" if is_home else "away"
    df[f"{prefix}_team_streak"] = streaks
    df[f"{prefix}_team_last_match_result"] = last_result
    df[f"{prefix}_team_points_last_3"] = last_points_3
    df[f"{prefix}_team_score_diff_last_5"] = score_diff_5

    return df


def add_season_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Essas features estatísticas acumuladas por temporada são essenciais pra entender a performance geral dos times ao longo do campeonato. 
    
    Vamos implementar isso da forma mais prática possível, considerando que seu DataFrame está ordenado por data e contém as colunas padrão: home_team, away_team, home_score, away_score, season.
    """
    df = df.copy()
    df = df.sort_values('datetime')  # importante garantir ordem cronológica

    # Dicionários para armazenar acumulados
    stats = {}

    # Listas para preencher depois no df
    home_goals_scored = []
    home_goals_conceded = []
    home_matches = []
    home_goal_diff = []

    away_goals_scored = []
    away_goals_conceded = []

    for idx, row in df.iterrows():
        season = row['season']
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        # Inicializa dicionário se ainda não tiver
        for team in [home, away]:
            if (season, team) not in stats:
                stats[(season, team)] = {
                    'scored': 0,
                    'conceded': 0,
                    'matches': 0,
                }

        # Adiciona as estatísticas antes do jogo atual
        home_stats = stats[(season, home)]
        away_stats = stats[(season, away)]

        home_goals_scored.append(home_stats['scored'])
        home_goals_conceded.append(home_stats['conceded'])
        home_matches.append(home_stats['matches'])
        home_goal_diff.append(home_stats['scored'] - home_stats['conceded'])

        away_goals_scored.append(away_stats['scored'])
        away_goals_conceded.append(away_stats['conceded'])

        # Atualiza estatísticas pós-jogo
        stats[(season, home)]['scored'] += home_score
        stats[(season, home)]['conceded'] += away_score
        stats[(season, home)]['matches'] += 1

        stats[(season, away)]['scored'] += away_score
        stats[(season, away)]['conceded'] += home_score
        stats[(season, away)]['matches'] += 1

    # Preenche no DataFrame
    df['home_team_total_goals_scored_season'] = home_goals_scored
    df['home_team_total_goals_conceded_season'] = home_goals_conceded
    df['home_team_matches_played_season'] = home_matches
    df['home_team_goal_diff_season'] = home_goal_diff

    df['away_team_total_goals_scored_season'] = away_goals_scored
    df['away_team_total_goals_conceded_season'] = away_goals_conceded

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    "early" (início): Janeiro a março.
    "mid" (meio): Abril a junho.
    "late" (fim): Julho a setembro.
    "final": Outubro a dezembro.
    """
    df = df.copy()

    # Adiciona o dia da semana
    df['match_day_of_week'] = df['datetime'].dt.day_name()

    # Define a fase da temporada (início, meio, final)
    def season_phase(row):
        if row['datetime'].month <= 3:  # Início da temporada
            return 'early'
        elif row['datetime'].month <= 6:  # Meio da temporada
            return 'mid'
        elif row['datetime'].month <= 9:  # Fim da temporada
            return 'late'
        else:
            return 'final'

    df['season_phase'] = df.apply(season_phase, axis=1)

    # Exemplo de como adicionar o número da rodada (aqui você precisaria de uma lógica adicional para isso)
    # Supondo que a coluna 'match_name' contenha a rodada, se disponível
    # df['round_number'] = df['match_name'].apply(lambda x: int(x.split(' ')[1]))  # Exemplo fictício

    return df

def update_team_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para atualizar o ranking dos times baseado nos resultados de cada partida.
    O ranking é reiniciado no início de cada temporada.
    """
    # Inicializa o dicionário para armazenar os rankings acumulados
    team_rankings = {}

    # Criar colunas de ranking para cada time
    df['home_team_ranking'] = 0
    df['away_team_ranking'] = 0

    # Itera sobre as partidas para calcular os rankings
    for index, row in df.iterrows():
        # Inicializa o ranking para a temporada, se não existir
        season = row['season']
        home_team = row['home_team']
        away_team = row['away_team']

        if season not in team_rankings:
            team_rankings[season] = {}

        # Inicializa o ranking dos times para a temporada
        if home_team not in team_rankings[season]:
            team_rankings[season][home_team] = 0
        if away_team not in team_rankings[season]:
            team_rankings[season][away_team] = 0
        
        # Atualiza os rankings baseados no resultado
        if row['result'] == 'H':  # Vitória do time da casa
            team_rankings[season][home_team] += 3
            team_rankings[season][away_team] += 0
        elif row['result'] == 'A':  # Vitória do time visitante
            team_rankings[season][home_team] += 0
            team_rankings[season][away_team] += 3
        elif row['result'] == 'D':  # Empate
            team_rankings[season][home_team] += 1
            team_rankings[season][away_team] += 1

        # Atribui o ranking ao DataFrame
        df.at[index, 'home_team_ranking'] = team_rankings[season][home_team]
        df.at[index, 'away_team_ranking'] = team_rankings[season][away_team]

    return df


def encode_categorical_features(df: pd.DataFrame, path_team_mapping: str) -> pd.DataFrame:
    df = df.copy()

    df = encode_teams(df, path_team_mapping)
    df = encode_generic(df)

    return df

# encode_categorical_features
def encode_teams(df: pd.DataFrame, path_team_mapping: str) -> pd.DataFrame:
    """
    Codifica os times usando Label Encoding e salva o mapeamento de-para em um arquivo JSON.

    :param df: DataFrame contendo os dados das partidas.
    :param path_team_mapping: Caminho do arquivo onde o mapeamento será salvo.
    :return: DataFrame com colunas adicionais para os times codificados.
    """
    df = df.copy()

    label_encoder = LabelEncoder()
    all_teams = pd.concat([df['home_team'], df['away_team']])
    label_encoder.fit(all_teams)

    # Codifica os times
    df['home_team_encoder'] = label_encoder.transform(df['home_team'])
    df['away_team_encoder'] = label_encoder.transform(df['away_team'])

    # Cria e salva o mapeamento
    json_team_mapping = {team: int(code) for team, code in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
    
    with open(path_team_mapping, 'w', encoding='utf-8') as f:
        json.dump(json_team_mapping, f, ensure_ascii=False, indent=4)

    return df

def encode_generic(df: pd.DataFrame) -> pd.DataFrame:
    # Cópia para não alterar original
    df_encoded = df.copy()

    # Label Encoding para resultado (vitória, empate, derrota)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df_encoded['result'])
    df_encoded["result"] = y

    # # One-hot encoding para as demais variáveis categóricas
    # categorical_cols = ['match_day_of_week', 'season_phase']
    # df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)

    return df_encoded
