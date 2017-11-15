import os,glob,fnmatch

os.chdir(r"C:/Users/pratik pc/Desktop/test")
for index, oldfile in enumerate(glob.glob("*.jpg")):
    newfile = '{}.jpg'.format(index+1)
    os.rename (oldfile,newfile)
