import os
import logging

# Configuração do logger
def setup_logger(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path =os.path.join(log_dir, log_file)
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    
    # Configuração de handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path, mode="w")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Adicionando handlers ao logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger