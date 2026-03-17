import hashlib
from git import Repo
import os
from utils.config import REPO_PATH

def get_repo_id(repo_url):
    return hashlib.md5(repo_url.encode()).hexdigest()[:10]

def clone_repo(repo_url):
    repo_id = get_repo_id(repo_url)
    target_path = os.path.join(REPO_PATH, repo_id)
    
    if os.path.exists(target_path):
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    Repo.clone_from(repo_url, target_path)
    return target_path
