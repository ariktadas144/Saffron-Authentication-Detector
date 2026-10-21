import zipfile, os
SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", "extracted", "runs"}
SKIP_EXT = {".zip"}
with zipfile.ZipFile("saffron_project.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1].lower() in SKIP_EXT: continue
            p = os.path.join(root, f)
            z.write(p, p[2:].replace("\\", "/"))
print("done")