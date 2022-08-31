import tarfile
import os
f = open("iiithws_ann.txt","r")
lines = f.readlines()
folder_name = []
for line in lines:
    val = "Images_90K_Normalized/"+line.split()[0]
    folder_name.append(val)
    print(line ," Appending ", val)

#os.mkdir("../../../extra/data/hrishikesh/iiit-hws")

print("list created")
dictio = {}
tar = tarfile.open("../../../extra/data/hrishikesh/iiit-hws.tar.gz")
subdir_and_files = []
for i,tarinfo in enumerate(tar.getmembers()):
    print(i)
    dictio[tarinfo.name] = tarinfo

for i,name in enumerate(folder_name):
    print(i)
    subdir_and_files.append(dictio[name])

tar.extractall(path = "../../../extra/data/hrishikesh/iiit-hws",members=subdir_and_files)

