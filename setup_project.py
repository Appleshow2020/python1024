# setup_project.py
import os
from classes.printing import *


# Defined project structure
project_structure = {
    "core": ["__init__.py", "config_manager.py", "data_broker.py", "process_manager.py"],
    "camera": ["__init__.py", "camera_process.py"],
    "detection": ["__init__.py", "detection_process.py"],
    "ui": ["__init__.py", "dashboard_process.py", "web_dashboard.py"],
    "ui/templates": ["dashboard.html"],
    "utils": ["__init__.py", "performance_monitor.py"],
    "classes": ["__init__.py"] # Empty folder for existing class files
}

# Top-level files
root_files = [
    "improved_main.py",
    "config.json",
    "requirements.txt",
    "README.md"
]

def create_project_structure():
    """Creates project directories and empty files."""
    printf("Starting project structure creation...", LT.info)
    # Create directories
    for directory, files in project_structure.items():
        try:
            os.makedirs(directory, exist_ok=True)
            printf(f"Directory created: {directory}", LT.info)
            # Create files in the directory
            for file in files:
                file_path = os.path.join(directory, file)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        # Leave __init__.py files empty
                        if not file.endswith('.py') or file == '__init__.py':
                            pass
                    printf(f"  -> File created: {file_path}", LT.info)
        except OSError as e:
            printf(f"Error occurred while creating directory '{directory}': {e}", LT.error)

    # Create top-level files
    for file in root_files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                 pass # Create as empty file
            printf(f"File created: {file}")

    printf("Project structure creation completed!", LT.success)

if __name__ == "__main__":
    create_project_structure()