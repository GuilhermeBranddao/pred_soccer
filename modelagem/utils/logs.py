import logging
import os

# Configuração de logs
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "main_train.log"

def setup_logger(name="main", log_dir=DEFAULT_LOG_DIR, log_file=DEFAULT_LOG_FILE, level=logging.DEBUG):
    """
    Configura e retorna um logger para registrar logs do programa.

    Parâmetros
    ----------
    name : str, opcional
        Nome do logger (padrão: "main").
    log_dir : str, opcional
        Diretório onde os logs serão armazenados (padrão: "logs").
    log_file : str, opcional
        Nome do arquivo de log (padrão: "main_train.log").
    level : logging.LEVEL, opcional
        Nível de log (padrão: logging.DEBUG).

    Retorna
    -------
    logging.Logger
        Objeto logger configurado.
    """

    os.makedirs(log_dir, exist_ok=True)  # Garante que a pasta de logs existe

    logger = logging.getLogger(name)

    # Limpa handlers antigos para evitar duplicação
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Formato dos logs
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(funcName)s | %(message)s"
    )

    # Log para arquivo
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Log para console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Criar um logger principal
logger = setup_logger()
