from drivers.download_csv_footebol_data import download_csv
from contracts.extract_contract import ExtractContract
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

class ExtractCsv:
    """
    Classe responsável por extrair dados de arquivos CSV localizados em um diretório específico.
    """

    def extract(self, path_base_download: Path = Path("database/csv")) -> ExtractContract:
        """
        Realiza o processo de extração dos arquivos CSV e retorna os dados como um contrato de extração.

        Args:
            path_base_download (Path): Diretório base onde os arquivos CSV estão localizados.

        Returns:
            ExtractContract: Objeto contendo os dados extraídos e a data de extração.

        Raises:
            FileNotFoundError: Se o diretório especificado não existir.
        """
        # Verifica se o diretório existe
        if not path_base_download.exists():
            path_base_download.mkdir(parents=True, exist_ok=True)
            print(f"O diretório {path_base_download} foi criado.")
            download_csv()
            # raise FileNotFoundError(f"O diretório {path_base_download} não existe.")

        # Carrega os arquivos CSV em um dicionário
        information_content = self._load_csv_files(path=path_base_download)

        # Cria o contrato de extração
        extract_contract = ExtractContract(
            information_content=information_content,
            extraction_date=datetime.now()
        )

        return extract_contract

    def _load_csv_files(self, path: Path) -> dict:
        """
        Carrega todos os arquivos CSV de um diretório em um dicionário.

        Args:
            path (Path): Diretório onde os arquivos CSV estão localizados.

        Returns:
            dict: Um dicionário onde as chaves são os nomes dos arquivos (sem extensão) 
                  e os valores são os DataFrames correspondentes.
        """
        information_content = {}
        csv_files = list(path.glob("*.csv"))  # Busca apenas arquivos com extensão .csv

        if not csv_files:
            print("Nenhum arquivo CSV encontrado no diretório.")
            return information_content

        for csv_file in csv_files:
            try:
                # Lê o arquivo CSV em um DataFrame
                df = pd.read_csv(csv_file)

                # Verifica se o DataFrame tem colunas válidas
                if df.empty or df.columns.isnull().any():
                    print(f"Arquivo ignorado (CSV vazio ou inválido): {csv_file.name}")
                    continue

                # Nome do arquivo sem extensão como chave no dicionário
                filename = csv_file.stem
                filename = filename.strip()
                information_content[filename] = df

                #print(f"Arquivo processado com sucesso: {csv_file.name}")
            except pd.errors.EmptyDataError:
                print(f"Arquivo CSV vazio ignorado: {csv_file.name}")
            except pd.errors.ParserError as e:
                print(f"Erro de parsing no arquivo {csv_file.name}: {e}")
            except Exception as e:
                print(f"Erro inesperado ao processar {csv_file.name}: {e}")

        return information_content
