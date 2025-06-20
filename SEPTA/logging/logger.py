import logging
import os 


LOG_FILE="log.logs"


logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

os.makedirs(logs_path,exist_ok=True)

LOGS_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format="[%(asctime)] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)