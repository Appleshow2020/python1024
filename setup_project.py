# setup_project.py
import os
from classes.printing import *

# 정의된 프로젝트 구조
project_structure = {
    "core": ["__init__.py", "config_manager.py", "data_broker.py", "process_manager.py"],
    "camera": ["__init__.py", "camera_process.py"],
    "detection": ["__init__.py", "detection_process.py"],
    "ui": ["__init__.py", "dashboard_process.py", "web_dashboard.py"],
    "ui/templates": ["dashboard.html"],
    "utils": ["__init__.py", "performance_monitor.py"],
    "classes": ["__init__.py"] # 기존 클래스 파일들을 위한 빈 폴더
}

# 최상위 파일들
root_files = [
    "improved_main.py",
    "config.json",
    "requirements.txt",
    "README.md"
]

def create_project_structure():
    """프로젝트 디렉토리와 빈 파일들을 생성합니다."""
    printf("프로젝트 구조 생성을 시작합니다...", LT.info)
    # 디렉토리 생성
    for directory, files in project_structure.items():
        try:
            os.makedirs(directory, exist_ok=True)
            printf(f"디렉토리 생성: {directory}",LT.info)
            # 해당 디렉토리 내 파일 생성
            for file in files:
                file_path = os.path.join(directory, file)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        # __init__.py 파일은 비워둡니다.
                        if not file.endswith('.py') or file == '__init__.py':
                            pass
                    printf(f"  -> 파일 생성: {file_path}", LT.info)
        except OSError as e:
            printf(f"'{directory}' 디렉토리 생성 중 오류 발생: {e}", LT.error)

    # 최상위 파일 생성
    for file in root_files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                 pass # 빈 파일로 생성
            printf(f"파일 생성: {file}")

    printf("\n프로젝트 구조 생성이 완료되었습니다!", LT.success)

if __name__ == "__main__":
    create_project_structure()