import numpy as np
from PIL import Image
import os

# works!
# combines 4 mnist images into one.
FOLDER_NAME = "lfw"
directory_in_str = os.path.abspath(os.curdir) + "\\" + FOLDER_NAME
main = Image.new("RGB", (640*2, 480*2))
img1 = Image.open(directory_in_str+"\\lfw_10_05_1.png")
img2 = Image.open(directory_in_str+"\\lfw_10_05_2.png")
img3 = Image.open(directory_in_str+"\\lfw_10_05_3.png")
img4 = Image.open(directory_in_str+"\\lfw_10_05_4.png")

main.paste(img1, (0,0))
main.paste(img2, (640,0))
main.paste(img3, (0,480))
main.paste(img4, (640,480))
main.save("main_lfw_10_05.png")
