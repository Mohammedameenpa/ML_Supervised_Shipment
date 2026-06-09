import joblib
import os

def save_object(file_name, obj):
    
    # Get project root folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    file_path = os.path.join(project_root, file_name)

    joblib.dump(obj, file_path)

    print(f"Saved successfully: {file_path}")





