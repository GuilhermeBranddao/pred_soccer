# main_trainer.py

import pandas as pd
import sqlite3

# from database.create_connection import create_connection
from etl.infra.database import create_connection
from modelagem.utils.logs import logger


def get_soccer_data(country:str = "Brazil") -> pd.DataFrame | None:
    """
    Recupera os dados de futebol do banco de dados SQLite com base no país especificado.

    Parameters
    ----------
    country : str, optional
        Nome do país para filtrar os dados. O padrão é "Brazil".

    Returns
    -------
    pd.DataFrame | None
        Um DataFrame contendo os dados do banco de dados, ou None em caso de erro.
    """
    database_path = "database/soccer_data.db"
    conn = create_connection(database_path)

    if conn is None:
        logger.error("Falha ao conectar ao banco de dados. Encerrando a operação.")
        return None

    try:
        query = "SELECT * FROM soccer_data WHERE country = ?"
        df = pd.read_sql_query(query, conn, params=(country,))

        if df.empty:
            logger.warning(f"Nenhum dado encontrado para o país: {country}")
            return None

        logger.info(f"Dados carregados com sucesso para o país: {country}")

        # outra maneira
        # cursor = conn.cursor()
        # cursor.execute(query)
        # tables = cursor.fetchall()
        return df

    except sqlite3.Error as e:
        logger.error(f"Erro ao executar consulta: {e}")
        return None
    finally:
        conn.close()
        logger.debug("Conexão com o banco de dados fechada.")