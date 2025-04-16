import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from modelagem.settings.config import Settings
config = Settings()

DEFAULT_LOG_FILE = "main_train.log"
DEFAULT_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(funcName)s | %(message)s"


def setup_logger(
    name: str = "main",
    log_dir: str = config.LOG_DIR,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = logging.DEBUG,
    log_format: str = DEFAULT_LOG_FORMAT,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3
) -> logging.Logger:
    """
    Configura e retorna um logger para registrar logs do programa com rotação de arquivos.

    Parâmetros:
        name (str): Nome do logger.
        log_dir (str): Diretório onde os logs serão armazenados.
        log_file (str): Nome do arquivo de log.
        level (int): Nível de log (ex: logging.DEBUG).
        log_format (str): Formato das mensagens de log.
        max_bytes (int): Tamanho máximo de cada arquivo antes da rotação.
        backup_count (int): Quantidade de backups a manter.

    Retorna:
        logging.Logger: Logger configurado.
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_path / log_file

    logger = logging.getLogger(name)

    # Evita recriar handlers se já estiver configurado
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(log_format)

        # Rotating File Handler
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Uso
logger = setup_logger()
logger.info("Logger configurado com sucesso.")
