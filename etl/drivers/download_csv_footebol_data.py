import os
import requests

def download_csv(
    base_url:str="https://www.football-data.co.uk/new",
    output_dir="database/csv",
    list_all_divisions:list=[None])->bool:

    # Lista de divisões e seus códigos
    list_all_divisions = {
        "Argentina": "ARG",
        "Áustria": "AUT",
        "Brasil": "BRA",
        "China": "CHN",
        "Dinamarca": "DNK",
        "Finlândia": "FIN",
        "Irlanda": "IRL",
        "Japão": "JPN",
        "México": "MEX",
        "Noruega": "NOR",
        "Polônia": "POL",
        "Romênia": "ROU",
        "Rússia": "RUS",
        "Suécia": "SWE",
        "Suíça": "CHE",
        "EUA": "USA"
    }

    # Criar o diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre as divisões e baixar os arquivos
    for country, code in list_all_divisions.items():
        file_url = os.path.join(base_url, f"{code}.csv")
        file_path = os.path.join(output_dir, f"{code}.csv")
        
        try:
            # Fazer o download do arquivo
            response = requests.get(file_url)
            response.raise_for_status()  # Verifica se houve erro no download
            
            # Salvar o arquivo no diretório especificado
            with open(file_path, "wb") as file:
                file.write(response.content)
            
            print(f"Download concluído: {file_path}")
        except requests.RequestException as e:
            print(f"Erro ao baixar {file_url}: {e}")
            return False
        
    return True
