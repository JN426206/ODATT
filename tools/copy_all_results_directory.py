# Tool for copy all dirs which contain "results" string in name (include subdirectories - recursive) to target directory.
import glob
import os
import shutil
import sys
from pathlib import Path

def scan(source, target):
    dirs = [(os.path.abspath(x) if os.path.isdir(x) and "results" in os.path.basename(x).lower() else None) for x in glob.glob(source+"/**/*", recursive=True)]
    numoffiles = len(dirs)
    print("Scanned number of directories: {}".format(numoffiles))
    os.makedirs(target, exist_ok=True)
    source = source if source[-1] != "/" else source[:-1]
    for mdir in dirs:
        if mdir is not None and mdir not in target and target not in mdir:
            dest_dir = os.path.join(target, mdir.replace(source+"/", ""))
            # print(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            for filename in os.listdir(mdir):
                file = os.path.join(mdir, filename)
                if os.path.isfile(file):
                    new_file = os.path.join(dest_dir, filename)
                    print(f"Copy {file} -> {new_file}")
                    shutil.copy(file, new_file)

if __name__ == "__main__":
    #print(len(sys.argv))
    if len(sys.argv)<2:
        print("Podaj jako 1 argument katalog źródłowy a jako 2 katalog przeszukiwany w którym zostaną usunięte pliki!")
        exit(0)

    scan(sys.argv[1],sys.argv[2])