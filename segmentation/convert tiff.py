import os
from PIL import Image

yourpath = os.getcwd()

list_dirs = [os.path.join(yourpath,'extra data/JSRT/Images/'), os.path.join(yourpath,'extra data/JSRT/Masks/')]                   
for directory in list_dirs:
    inputdir = directory
    outdir = directory
    
    os.chdir(directory)
    
    test_list = [ f for f in  os.listdir(inputdir)]

    for f in test_list:   # remove "[:10]" to convert all images 
         if(f.endswith(".tif")):
            im = Image.open(os.path.join(inputdir, f)) # read tiff image
            #cv2.imwrite(outdir + f.replace('.tif','.png'),img) # write png image
            im = im.convert("L")
            im.thumbnail(im.size)
            output = outdir + f.replace('.tif','.png')
            print(output)
            im.save(output, "png", quality=100)
            os.remove(f)