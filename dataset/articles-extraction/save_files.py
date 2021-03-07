import os
from itertools import groupby
from operator import itemgetter

path = "."

paths = []
for f in os.scandir(path):
    if f.is_dir():
        dir_name = f.path
        if ".com" in dir_name:
            paths.append(dir_name)

# print(paths)


for path in paths:
    SEARCH_PATH = path




    all_files = []

    for root, dirs, files in os.walk(SEARCH_PATH):
        for file in files:
            relativePath = os.path.relpath(root, SEARCH_PATH)
            if relativePath == ".":
                relativePath = ""
            all_files.append( (relativePath.count(os.path.sep),  relativePath, file )
            )

    # all_files.sort(reverse=True)

    for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
        
        for file in files:
            print(SEARCH_PATH + '/' + folder + '/' + file[2])