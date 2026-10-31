"""
NeuroScan-AI: Automated GitHub Repository Creation and Deployment Script
Creates the public repository 'neuroscan-brain-tumor-detection' on sumithalanke46
and pushes all source code, assets, notebook, and the 15-page academic report.
"""

import os
import subprocess
import urllib.request
import json

REPO_NAME = "neuroscan-brain-tumor-detection"
REPO_DESC = "NeuroScan-AI: Multiclass Brain Tumor MRI Classification via Deep Transfer Learning (PyTorch, MobileNetV2, Streamlit CAD Portal). Achieves 97.00% test accuracy."


def get_github_token():
    """Extracts GitHub OAuth / PAT token from Windows Git Credential Manager."""
    proc = subprocess.Popen(
        ['git', 'credential', 'fill'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, _ = proc.communicate("protocol=https\nhost=github.com\n\n")
    username = None
    token = None
    for line in out.splitlines():
        if line.startswith('username='):
            username = line.split('=', 1)[1]
        elif line.startswith('password='):
            token = line.split('=', 1)[1]
    return username, token


def create_github_repo(token: str, username: str, repo_name: str, description: str):
    """Creates a public repository under the user account using GitHub REST API."""
    url = "https://api.github.com/user/repos"
    payload = json.dumps({
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "NeuroScan-AI-Uploader",
        "Content-Type": "application/json"
    })

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            print(f"Successfully created GitHub repository: {data.get('html_url')}")
            return data.get('html_url')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        if e.code == 422 and "already exists" in error_body:
            print(f"Repository '{repo_name}' already exists on GitHub under {username}. Proceeding with push.")
            return f"https://github.com/{username}/{repo_name}"
        else:
            raise RuntimeError(f"GitHub API error ({e.code}): {error_body}")


def deploy_repository():
    print("=" * 65)
    print("NeuroScan-AI: GitHub Deployment Initiated")
    print("=" * 65)

    username, token = get_github_token()
    if not token or not username:
        raise ValueError("Could not retrieve GitHub credentials from Git Credential Manager.")

    print(f"Authenticated as GitHub user: {username}")
    repo_url = create_github_repo(token, username, REPO_NAME, REPO_DESC)

    # Git Operations
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(f"Working Directory: {cwd}")

    # Remove temporary scripts before committing
    for tmp in ["extract_logo.py", "verify_report.py"]:
        p = os.path.join(cwd, tmp)
        if os.path.exists(p):
            os.remove(p)

    cmds = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.name", username],
        ["git", "config", "user.email", f"{username}@gmail.com"],
        ["git", "add", "."],
        ["git", "status"],
        ["git", "commit", "-m", "Initial release: NeuroScan-AI Deep Transfer Learning Brain Tumor MRI Classification system with Streamlit CAD app, 15-page academic report, evaluation metrics, and PyTorch model"],
    ]

    for cmd in cmds:
        print(f"Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr and "warning" not in res.stderr.lower():
            print(f"Stderr: {res.stderr}")

    # Set authenticated remote URL
    remote_auth_url = f"https://{token}@github.com/{username}/{REPO_NAME}.git"
    
    # Check if origin exists
    check_remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True, text=True)
    if check_remote.returncode == 0:
        subprocess.run(["git", "remote", "set-url", "origin", remote_auth_url], cwd=cwd, check=True)
    else:
        subprocess.run(["git", "remote", "add", "origin", remote_auth_url], cwd=cwd, check=True)

    print(f"Pushing to origin main...")
    push_res = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=cwd, capture_output=True, text=True)
    print(push_res.stdout)
    if push_res.stderr:
        print(push_res.stderr)

    if push_res.returncode == 0:
        # Reset remote URL to clean public HTTPS URL without token
        clean_url = f"https://github.com/{username}/{REPO_NAME}.git"
        subprocess.run(["git", "remote", "set-url", "origin", clean_url], cwd=cwd, check=True)
        print("=" * 65)
        print(f"DEPLOYMENT SUCCESSFUL!")
        print(f"Public GitHub Repository: https://github.com/{username}/{REPO_NAME}")
        print("=" * 65)
        return f"https://github.com/{username}/{REPO_NAME}"
    else:
        raise RuntimeError(f"Git push failed with code {push_res.returncode}")


if __name__ == "__main__":
    deploy_repository()
