import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np

window = tkinter.Tk()

cv_img = cv2.imread(r'D:\Karthika_DIS\Project_three\pic_1.jpg',cv2.COLOR_BGR2RGB)
x=cv_img
height, width, no_channels = x.shape
canvas = tkinter.Canvas(window, width = width, height = height)
canvas.pack()

photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(x))

canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
h,w=x.shape[:2]

r = cv2.selectROI(x)
imCrop = x[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
a=imCrop.shape
l,m=imCrop.shape[:2]
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imCrop))

canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
mask = np.zeros((h,w,3))
mask[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]=255
new_image=mask.astype(np.uint8)
photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(new_image))
canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
window.mainloop()
