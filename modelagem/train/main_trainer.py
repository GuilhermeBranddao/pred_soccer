# main_trainer.py

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import sqlite3
from pathlib import Path
from modelagem.feature_eng.match_analysis import get_storage_ranks, create_main_cols

# from database.create_connection import create_connection
from etl.infra.database import create_connection
from modelagem.utils.preprocessing.engine import base_pre_processing
from modelagem.utils.logs import logger
from modelagem.train import model_trainer

# Definindo diretórios base
BASE_DIR = os.path.dirname(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(BASE_DIR, 'feature_eng', 'data')
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), 'database')
LOG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'logs')

# logger = setup_logger(LOG_DIR, "main_train.log")

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


if __name__ == "__main__":
    df = get_soccer_data()
    success = base_pre_processing(df)
    if success:
        logger.info("Pré-processamento concluído com sucesso!")
    else:
        logger.error("Erro no pré-processamento.")

    df = pd.read_csv(os.path.join(DATA_DIR, 'ft_df.csv'))
    logger.info("Iniciando treinamento")
    model_trainer.main(df)
