from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "rahuldhane/lending-club-credit-risk"

create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
print(f"Space created/confirmed: {repo_id}")

files = ["app.py", "deployment_artifacts.joblib", "requirements.txt", "README.md"]
for f in files:
    api.upload_file(
        path_or_fileobj=f"D:/Lending/{f}",
        path_in_repo=f,
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"Uploaded: {f}")

print(f"\nSpace URL: https://huggingface.co/spaces/{repo_id}")
