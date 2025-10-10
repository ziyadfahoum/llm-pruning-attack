# copy files recursively under specified directory
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir_under_misc", type=str, required=True)
args = parser.parse_args()

src = Path("misc") / args.dir_under_misc
dst = Path(args.dir_under_misc)

for file in src.rglob("*"):
    if file.is_file():
        target_path = dst / file.relative_to(src)
        print(f"{file} \n\t---> {target_path}")
        assert target_path.parent.exists(), f"Target directory does not exist: {target_path.parent}"
        assert target_path.exists(), f"Target file does not exist: {target_path}. This command is intended to 'replace' files."
        shutil.copy2(file, target_path)
