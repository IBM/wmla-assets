#!/usr/bin/env python3

import os
os.system("mkdir -p csv/test csv/train")
os.system("cp ./imdb.vocab csv/ctx.txt")

for purpose in ["test", "train"]:
    with open("./csv/" + purpose + "/" + purpose + ".csv", "w", encoding="utf8") as csv_file:
        for myclass in ["pos", "neg"]:
            for (dirpath, dirnames, filenames) in os.walk("./" + purpose + "/" + myclass + "/"):
                for myfile in filenames:
                    with open(os.sep.join([dirpath, myfile])) as file:
                        csv_file.write(myclass + "\\")
                        for lines in file:
                            csv_file.write(lines.strip())
                        csv_file.write("\n")

os.system("tar -czvf csv.tgz csv/")