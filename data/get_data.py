import shutil
from pathlib import Path

import kagglehub

dest = Path(__file__).parent

source = Path(kagglehub.dataset_download("siddhrajthakor/football-manager-2023-dataset"))

for item in source.iterdir():
    target = dest / item.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)

print("Path to dataset files:", dest)