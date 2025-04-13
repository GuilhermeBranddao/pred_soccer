from pathlib import Path
import pandas as pd
from etl.contracts.extract_contract import ExtractContract
from etl.contracts.transform_contract import TransformContract

from etl.infra.database import create_connection, filter_new_data

class TransformCsv:
    """
    Classe responsável por carregar os dados extraídos e processados para um diretório ou sistema de armazenamento.
    """
    def __init__(self, conn):
        self.conn = conn

    def transform(self, extract_contract: ExtractContract):
        """
        Realiza a transformação dos dados e os insere ou atualiza no banco SQLite.

        Args:
            information_content (dict): Dicionário contendo os dados dos CSVs.
        """

        transform_data = self.__filter_and_transform_data(extract_contract)
            
        return TransformContract(
            transform_data=transform_data
        )

    def __filter_and_transform_data(self, extract_contract:ExtractContract):
        transform_data = {}

        list_keys = extract_contract.information_content.keys()
        for key in list_keys:
            df = extract_contract.information_content[key]
            if df.empty:
                continue
            
            df.columns = [columns.lower() for columns in df.columns]
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:%M')
            df.drop(columns=["date", "time"], inplace=True)
            
            df.rename(columns={"home":"home_team", "away":"away_team", 
                               "hg":"home_score", "ag":"away_score", 
                               "res":"result"}, inplace=True)
            
            df = filter_new_data(conn=self.conn, df=df)

            transform_data[key] = df

        return transform_data