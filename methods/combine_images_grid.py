import numpy as np
from PIL import Image
import os

# works!
# combines 4 mnist images into one.

os.chdir("..")
directory_in_str = os.path.abspath(os.curdir) + "\\report"
main = Image.new("RGB", (640*2, 480*2))
img1 = Image.open(directory_in_str+"\\mnist1.png")
img2 = Image.open(directory_in_str+"\\mnist2.png")
img3 = Image.open(directory_in_str+"\\mnist3.png")
img4 = Image.open(directory_in_str+"\\mnist4.png")

main.paste(img1, (0,0))
main.paste(img2, (640,0))
main.paste(img3, (0,480))
main.paste(img4, (640,480))
main.save("main_mnist.png")
