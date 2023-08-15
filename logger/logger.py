import logging
import logging.config
from pathlib import Path
from utils import read_json

"""
이 코드는 로깅(logging) 설정을 초기화하는 함수인 setup_logging을 정의하는 파이썬 스크립트입니다. 
로깅은 프로그램의 실행 중에 발생하는 이벤트와 정보를 기록하는데 사용되며, 디버깅 및 모니터링에 유용합니다. 
"""
def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file(): # 로그 설정 파일이 존재하면
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items(): # config의 모든 핸들러를 순회
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename']) # 해당 핸들러의 'filename'(로그 파일 저장 경로)을 설정된 저장 디렉토리와 결합하여 업데이트합니다.

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)