import numpy as np
from PIL import Image
import os

# combines 6  images into one.
# can generalise to any number of images, just 
# init original img accordingly.
FOLDER_NAME = "flow_procedure"
directory_in_str = os.path.abspath(os.curdir) + "\\" + FOLDER_NAME
main = Image.new("RGB", (4000*2, 2250*3))
img1 = Image.open(directory_in_str+"\\flow_1.jpg")
img2 = Image.open(directory_in_str+"\\flow_2.jpg")
img3 = Image.open(directory_in_str+"\\flow_3.jpg")
img4 = Image.open(directory_in_str+"\\flow_4.jpg")
img5 = Image.open(directory_in_str+"\\flow_5.jpg")
img6 = Image.open(directory_in_str+"\\flow_6.jpg")

main.paste(img1, (0,0))
main.paste(img2, (4000,0))
main.paste(img3, (0,2250))
main.paste(img4, (4000,2250))
main.paste(img5, (0,2250*2))
main.paste(img6, (4000,2250*2))
main.save("main_flow_procedure.png")
