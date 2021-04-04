import os 
import hashlib 
from typing import Optional

def check_integrity(file_path: str, file_md5: Optional[str] = None) -> bool:
    if not os.path.exists(file_path):
        return False 
    if file_md5 is None:
        return True
    return file_md5 == hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    
