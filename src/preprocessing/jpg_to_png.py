import os
from PIL import Image

def folder(indir: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for root, dirs, files in os.walk(indir):
        for file in files:
            if file.lower().endswith('.jpg'):
                img = Image.open(os.path.join(root, file))
                out_file = os.path.splitext(file)[0] + '.png'
                img.save(os.path.join(outdir, out_file))

#
#   This sample code once uncommented and run will convert all jpg files in src/images/raw_jpg
#   into png files and put them into src/images/raw_png
#   BASE = os.path.dirname(os.path.abspath(__file__))
#   folder(
#       os.path.join(BASE, "../images/raw_jpg"),
#       os.path.join(BASE, "../images/raw_png")
#   )
#
