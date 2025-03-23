import pytest
from pathlib import Path
from datetime import datetime
from etl.stages.extract.extract_csv import ExtractCsv
from etl.contracts.extract_contract import ExtractContract
import pandas as pd
import os

@pytest.fixture
def create_test_directory(tmp_path):
    """
    Cria um diretório temporário para armazenar arquivos CSV durante os testes.
    """
    test_dir = tmp_path / "csv_test"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def populate_test_directory(create_test_directory):
    """
    Adiciona arquivos CSV ao diretório temporário.
    """
    test_dir = create_test_directory

    # Cria arquivos CSV de exemplo
    file_1 = test_dir / "file1.csv"
    file_2 = test_dir / "file2.csv"
    file_empty = test_dir / "empty.csv"

    # Dados de exemplo para os arquivos CSV
    file_1.write_text("col1,col2\n1,2\n3,4")
    file_2.write_text("colA,colB\nA,B\nC,D")
    file_empty.write_text("")  # Arquivo vazio

    return test_dir

def test_directory_not_exists():
    """
    Testa se o método 'extract' levanta um erro quando o diretório não existe.
    """
    extract_csv = ExtractCsv()
    non_existing_dir = Path("non_existing_dir")

    with pytest.raises(FileNotFoundError) as excinfo:
        extract_csv.extract(non_existing_dir)

    assert f"O diretório {non_existing_dir} não existe." in str(excinfo.value)

def test_no_csv_files(create_test_directory):
    """
    Testa o comportamento do método quando não há arquivos CSV no diretório.
    """
    extract_csv = ExtractCsv()
    result = extract_csv.extract(create_test_directory)

    # O resultado deve ser um contrato vazio
    assert isinstance(result, ExtractContract)
    assert result.information_content == {}
    assert isinstance(result.extraction_date, datetime)

def test_extract_valid_csv_files(populate_test_directory):
    """
    Testa se arquivos CSV válidos são processados corretamente.
    """
    extract_csv = ExtractCsv()
    result = extract_csv.extract(populate_test_directory)

    # Verifica se o resultado é do tipo ExtractContract
    assert isinstance(result, ExtractContract)

    # Verifica o conteúdo do dicionário
    assert "file1" in result.information_content
    assert "file2" in result.information_content
    assert isinstance(result.information_content["file1"], pd.DataFrame)
    assert isinstance(result.information_content["file2"], pd.DataFrame)

    # Verifica os valores do DataFrame
    assert result.information_content["file1"].shape == (2, 2)
    assert list(result.information_content["file1"].columns) == ["col1", "col2"]

def test_ignore_empty_csv(populate_test_directory):
    """
    Testa se arquivos CSV vazios são ignorados.
    """
    extract_csv = ExtractCsv()
    result = extract_csv.extract(populate_test_directory)

    # Verifica que o arquivo vazio não está no resultado
    assert "empty" not in result.information_content

def test_error_handling_invalid_csv(create_test_directory):
    """
    Testa se erros de parsing são tratados corretamente.
    """
    test_dir = create_test_directory

    # Cria um arquivo CSV inválido
    invalid_csv = test_dir / "invalid.csv"
    #invalid_csv.write_text("this,is\nnot,a,valid,csv")

    extract_csv = ExtractCsv()
    result = extract_csv.extract(test_dir)

    # Arquivo inválido deve ser ignorado
    assert "invalid" not in result.information_content
