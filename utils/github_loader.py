import hashlib
from git import Repo
import os
from utils.config import UPLOAD_DIR

def get_repo_id(repo_url):
    return hashlib.md5(repo_url.encode()).hexdigest()[:10]

def parse_github_url(url):
    """
    Parses a GitHub URL to extract base repository, branch, and subdirectory.
    Example: https://github.com/user/repo/tree/main/subdir
    Returns: (base_url, branch, subdir)
    """
    if "/tree/" in url:
        base_url = url.split("/tree/")[0]
        rest = url.split("/tree/")[1].split("/")
        branch = rest[0]
        subdir = "/".join(rest[1:])
        return base_url, branch, subdir
    return url, None, ""

def clone_repo(repo_url):
    repo_id = get_repo_id(repo_url)
    target_path = os.path.join(UPLOAD_DIR, repo_id)
    
    if os.path.exists(target_path):
        # We need to return both the base path and the subdir for the chain to work
        base_url, branch, subdir = parse_github_url(repo_url)
        return os.path.join(target_path, subdir)
    
    os.makedirs(target_path, exist_ok=True)
    
    base_url, branch, subdir = parse_github_url(repo_url)
    
    if branch:
        Repo.clone_from(base_url, target_path, branch=branch)
    else:
        Repo.clone_from(base_url, target_path)
        
    return os.path.join(target_path, subdir)
