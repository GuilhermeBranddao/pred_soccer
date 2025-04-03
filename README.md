Passo a passo de como usar o projeto

Passo 1: Crie um ambiente virtual 
$ python3.12 -m venv venv_3_12

Passo 2: Ative o ambiente virtual no terminal
$ venv_3_13_2\Scripts\activate

Passo 3: Instalar bibliotecas
$ pip install -r requirements.txt

Passo 4: Rode o ETL para prepara a base de dados
python -m pred_soccer.etl.main

Passo 5: 


# Passo x: Treine os modelos
python -m modelagem.train.main_trainer
