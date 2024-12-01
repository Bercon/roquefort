import os
import shutil
from pathlib import Path
import re

def main(args):
    OUTPUT_DIR = "build"
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    Path(os.path.join(OUTPUT_DIR, "src")).mkdir(parents=True, exist_ok=True)

    with open("index.html") as f, \
        open(os.path.join(OUTPUT_DIR, "index.html"), "w") as of, \
        open(os.path.join(OUTPUT_DIR, "src/main.js"), "w") as mf:
        for l in f.readlines():
            if not l.strip(): continue # Skip empty lines
            match = re.search(r'<script\s+src="([^"]+)"', l)
            if match:
                jsSrc = match.group(1)
                with open(jsSrc) as js:
                    mf.write(js.read())
                    mf.write(os.linesep)
                if (jsSrc == "src/main.js"):
                    of.write(l)
            else:
                of.write(l)

    shutil.copytree(os.path.join("src", "font"), os.path.join(OUTPUT_DIR, "src", "font"))
    shutil.copyfile(os.path.join("src", "style.css"), os.path.join(OUTPUT_DIR, "src", "style.css"))

    print("Build done!")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    args = p.parse_args()
    main(args)
