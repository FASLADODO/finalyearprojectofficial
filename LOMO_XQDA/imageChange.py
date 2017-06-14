import numpy as np
import glob
import os, sys
import Image

MIRROR=1
ROTATE_RIGHT=2
mode=ROTATE_RIGHT
if(mode==MIRROR):
	ext='mirrored/'
elif(mode==ROTATE_RIGHT):
	ext='rotated_right/'


files=['./images/'+i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/LOMO_XQDA/images") if i.find(".png")!=-1 ]
filesOut=['./images/'+ext+i for i in os.listdir("/home/rs6713/Documents/finalyearprojectofficial/LOMO_XQDA/images") if i.find(".png")!=-1 ]
for idx,f in enumerate(files):
    print f
    try:
        im = Image.open(f)
        print im.format, im.size, im.mode
	if(mode==MIRROR):
        	out=im.transpose(Image.FLIP_LEFT_RIGHT)
	if(mode==ROTATE_RIGHT):
		out= im.rotate(5)
	out.save(filesOut[idx], "PNG")
    except IOError:
	print 'ioerror'
        pass
