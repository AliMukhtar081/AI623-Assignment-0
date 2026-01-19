
import sys
from pathlib import Path

def setup_project_root():
  
    
    current_path = Path(__file__).resolve()
   
    for parent in current_path.parents:
        if (parent / 'src').exists() and (parent / 'src').is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
            
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
