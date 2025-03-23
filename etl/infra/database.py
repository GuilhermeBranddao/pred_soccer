import sqlite3
import pandas as pd
from pandas import DataFrame
import os

def create_connection(db_path):
    """Cria a conexão com o banco de dados SQLite."""
    connection = None
    try:
        
        is_exists = os.path.exists(db_path)

        connection = sqlite3.connect(db_path)

        if not is_exists:
            create_table(connection)

        print("Conexão SQLite estabelecida.")
    except sqlite3.Error as e:
        print(f"Erro ao conectar ao SQLite: {e}")
    return connection

def close_connection(connection):
    """Fecha a conexão com o banco de dados."""
    if connection:
        connection.close()
        print("Conexão SQLite encerrada.")

def create_table(connection):
    """Cria a tabela no banco de dados se não existir."""
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS soccer_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_score INTEGER NOT NULL,
            away_score INTEGER NOT NULL,
            result TEXT NOT NULL,
            psch REAL,
            pscd REAL,
            psca REAL,
            maxch REAL,
            maxcd REAL,
            maxca REAL,
            avgch REAL,
            avgcd REAL,
            avgca REAL,
            bfech REAL,
            bfecd REAL,
            datetime DATETIME NOT NULL,
            hash TEXT NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(hash)
        )
    """
    try:
        with connection:
            connection.execute(create_table_sql)
        print("Tabela 'soccer_data' criada com sucesso.")
    except sqlite3.Error as e:
        print(f"Erro ao criar tabela: {e}")

def __insert_data(connection, df: DataFrame):
    """Insere ou atualiza dados no banco de dados a partir do DataFrame."""
    # Substituindo valores nulos por None para o SQLite
    df = df.where(pd.notna(df), None)
    

    df.dropna(subset=['away_team', 'home_score', 'result'], inplace=True)

    # Convertendo a coluna 'Datetime' para string no formato ISO 8601
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Gerar a coluna de hash
    df['hash'] = pd.util.hash_pandas_object(df[['country', 'league', 'home_team', 'away_team', 'datetime']].astype(str), index=False).astype(str)

    # Convertendo o DataFrame em uma lista de dicionários
    data = df.to_dict('records')

    insert_sql = """
    INSERT OR REPLACE INTO soccer_data (
        country, league, season, home_team, away_team, home_score, away_score, result, psch, pscd, psca,
        maxch, maxcd, maxca, avgch, avgcd, avgca, bfech, bfecd, datetime, hash
    ) VALUES (
        :country, :league, :season, :home_team, :away_team, :home_score, :away_score, :result, :psch, :pscd, :psca,
        :maxch, :maxcd, :maxca, :avgch, :avgcd, :avgca, :bfech, :bfecd, :datetime, :hash
    )
    """
    
    try:
        with connection:
            connection.executemany(insert_sql, data)
        print(f"{len(data)} registros inseridos/atualizados com sucesso.")
    except sqlite3.Error as e:
        print(f"Erro ao inserir dados: {e}")

def insert_new_data(conn, df):
    """
    Insere os dados no banco de dados se não existirem.

    Args:
        conn: Conexão com o banco de dados SQLite.
        df (DataFrame): Dados a serem inseridos no banco de dados.
    """
    # Filtra os dados que já existem no banco
    df = filter_new_data(conn=conn, df=df)
    
    # Se o jogo não existir, insere no banco
    if not df.empty:
        __insert_data(conn, df)
        #print(f"{len(df)} registros novos inseridos.")
    else:
        print("Nenhum novo dado a ser inserido.")

def filter_new_data(conn, df:DataFrame):
    """
    Filtra os dados que ainda não existem no banco de dados.
    
    Args:
        conn: Conexão com o banco de dados SQLite.
        df (DataFrame): Dados a serem verificados.
        
    Returns:
        DataFrame: Dados filtrados contendo apenas os registros que não existem no banco.
    """

    # Gerar hashes para os dados do DataFrame
    df['hash'] = pd.util.hash_pandas_object(df[['country', 'league', 'home_team', 'away_team', 'datetime']].astype(str), index=False).astype(str)
    
    # Recuperar hashes já existentes no banco
    query = "SELECT hash FROM soccer_data"
    existing_hashes = pd.read_sql_query(query, conn)['hash'].tolist()

    # Filtrar o DataFrame para manter apenas os registros com hashes não existentes
    new_data = df[~df['hash'].isin(existing_hashes)]
    return new_data

# Exemplo de uso:
# connection = create_connection('path_to_your_db.db')
# if connection:
#     create_table(connection)
#     insert_new_data(connection, df)
#     close_connection(connection)
