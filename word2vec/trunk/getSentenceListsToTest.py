import os
files=[i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/LOMO_XQDA/data/sentences") if i.find("norm3")!=-1 and i.find("size400")!=-1 and i.find("mode2")!=-1 and i.find("trainLevel")==-1 ]

for f in range(len(files)):
	print files[f]
	files[f]=files[f].replace('.mat','.txt')
print files
