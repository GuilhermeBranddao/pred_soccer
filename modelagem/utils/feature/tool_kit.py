import pandas as pd
import os
from modelagem.utils.logs import logger
from pathlib import Path
from modelagem.utils.feedback import FunctionResult
# from modelagem.feature_eng.match_analysis import get_storage_ranks
from modelagem.utils.feature.encode import (
    encode_categorical_features, 
)
from typing import List
from modelagem.settings.config import Settings
config = Settings()


def drop_unwanted_features(df: pd.DataFrame, drop_columns:list=None) -> pd.DataFrame:

    drop_columns = drop_columns or []
    return df.drop(columns=[col for col in drop_columns if col in df.columns])

def check_missing_columns(columns_request: List[str], 
                            df_columns: List[str]) -> FunctionResult:
    """
    Verifica se todas as colunas requisitadas estão presentes no DataFrame (case-insensitive).
    
    Parâmetros:
        columns_request (List[str]): Lista de colunas esperadas.
        df_columns (List[str]): Lista de colunas disponíveis no DataFrame.
        encode_name (str): Nome do processo/etapa onde a verificação está sendo feita (opcional).
    
    Retorno:
        None. Imprime mensagem de erro se houver colunas ausentes.
    """
    # Normaliza para comparação case-insensitive
    request_set = {col.lower() for col in columns_request}
    df_set = {col.lower() for col in df_columns}

    # Verifica colunas ausentes
    missing = request_set - df_set

    if missing:
        missing_str = ", ".join(sorted(missing))
        logger.debug(f"❌ As seguintes colunas estão ausentes no DataFrame: [{missing_str}].")
        return FunctionResult(False, f"As seguintes colunas estão ausentes no DataFrame: [{missing_str}].", data=pd.DataFrame()) 
    else:
        logger.debug(f"✅ Todas as colunas necessárias para o encode estão presentes.")
        return FunctionResult(True, "Todas as colunas necessárias para o encode estão presentes.", data=pd.DataFrame()) 

def prep_data_to_save(df: pd.DataFrame, path_encoder: str, columns_request:list[str]) -> pd.DataFrame:
    # Se ok deve salvar o que for necessario para realizar predições
    # encodes, features, ....

    # Codifica os times
    df = encode_categorical_features(df, path_save_encoder=path_encoder)
    df.fillna(0, inplace=True)
    # Drop de colunas descenessarias

    # df = drop_unwanted_features(df, drop_columns=drop_columns)

    err = check_missing_columns(df_columns=df.columns, 
                          columns_request=columns_request)

    if not err.success:
        raise ValueError(f"Colunas faltando no DataFrame: {err.message}")

    df = df[columns_request]
    

    logger.debug("Salvando o DataFrame resultante.")
    os.makedirs(config.FT_DIR, exist_ok=True)
    output_path = os.path.join(config.FT_DIR, 'ft_df.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Feature DataFrame salvo em {output_path}")
    return True, df

