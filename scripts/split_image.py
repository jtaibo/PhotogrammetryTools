import os
import argparse

def splitImage(input_image):

    se = os.path.splitext(input_image)
    base_name = se[0]
    extension = se[1]
    if extension == ".insp":
        extension = ".jpg"

    cmd = "convert " + input_image + " -gravity West -crop 50%x100%+0+0 " + base_name + "_left" + extension
    os.system(cmd)
    cmd = "convert " + input_image + " -gravity East -crop 50%x100%+0+0 " + base_name + "_right" + extension
    os.system(cmd)



parser = argparse.ArgumentParser("split_image")
parser.add_argument("input_image", help="Path for the image to split in halves", default="")

args = parser.parse_args()



splitImage(args.input_image)

#convert $1 -gravity East -crop 50%x100%+0+0 right%04d.png
#convert $1 -gravity West -crop 50%x100%+0+0 left%04d.png
