import cv2
import numpy as np
from PIL import Image



print('Enter 1 for lena.png in RGB mode\nEnter 2 for lena.png in BW mode\nEnter 3 for pic_1.jpg in RGB mode\nEnter 4 for pic_1.jpg in BW mode')
val=int(input("ENTER CHOICE OF IMAGE: "))

#### Which Image?
if val ==1 :
    pic_1= cv2.imread(r'D:\Karthika_DIS\Project_three\lena.png',1)
    h_1, w_1 = pic_1.shape[:2]
    ip_img_usr = np.copy(pic_1) 

if val == 2 :
    pic_1= cv2.imread(r'D:\Karthika_DIS\Project_three\lena.png',1)
    h_1, w_1 = pic_1.shape[:2]
    gray = cv2.cvtColor(pic_1, cv2.COLOR_BGR2GRAY)
    ip_img_usr = np.copy(gray)
    
if val == 3:
    pic_1= cv2.imread(r'D:\Karthika_DIS\Project_three\pic_1.jpg',1)
    h_1, w_1 = pic_1.shape[:2]
    ip_img_usr = np.copy(pic_1) 

if val == 4:
    pic_1= cv2.imread(r'D:\Karthika_DIS\Project_three\pic_1.jpg',1)
    h_1, w_1 = pic_1.shape[:2]
    gray = cv2.cvtColor(pic_1, cv2.COLOR_BGR2GRAY)
    ip_img_usr = np.copy(gray)
   


print('Please enter number of layers:')
num_layers=3
h,w=pic_1.shape[:2]
count=[]

def pad_clr(image):
    h, w = image.shape[:2]
    fr_copy = image[0:1, ::1, ::]
    a = fr_copy.shape
    lr_copy = image[h - 1:h, ::1, ::]
    lc_copy = image[::1, w - 1:w, ::]
    fc_copy = image[::1, 0:1, ::]
    new_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2, 3))
    pad_main = np.copy(image[0:h, 0:w, ::])
    l, b = new_image.shape[:2]
    new_image[1:h + 1, 1:w + 1, ::] = pad_main
    new_image[0:1, 1:w + 1, ::] = lr_copy
    new_image[l - 1:l:1, 1:b - 1, ::] = fr_copy
    new_image = np.array(new_image, dtype=np.uint8)
    nh, nw = new_image.shape[:2]
    return new_image
    
def pad_bw(image):
    h, w = image.shape[:2]
    fr_copy = image[0:1, ::1]
    lr_copy = image[h - 1:h, ::1]
    lc_copy = image[::1, w - 1:w]
    fc_copy = image[::1, 0:1]
    new_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    pad_main = np.copy(image[0:h, 0:w])
    l, b = new_image.shape[:2]
    new_image[1:h + 1, 1:w + 1] = pad_main
    new_image[0:1, 1:w + 1] = lr_copy
    new_image[l - 1:l:1, 1:b - 1] = fr_copy
    new_image[1:l - 1, b - 1:b] = fc_copy
    new_image[1:l - 1, 0:1] = lc_copy
    new_image = np.array(new_image, dtype=np.uint8)
    nh, nw = new_image.shape[:2]
    return new_image

def ComputePyr(ip_img_usr,num_layers):
    if len(ip_img_usr.shape) == 2:
        h, w = ip_img_usr.shape[:2]
        fr_copy = ip_img_usr[0:1, ::1]
        lr_copy = ip_img_usr[h - 1:h, ::1]
        lc_copy = ip_img_usr[::1, w - 1:w]
        fc_copy = ip_img_usr[::1, 0:1]
        new_image = np.zeros((h_1 + 2, w_1 + 2))
        pad_main = np.copy(ip_img_usr[0:h, 0:w])
        l, b = new_image.shape[:2]
        new_image[1:h + 1, 1:w + 1] = pad_main
        # first row
        new_image[0:1, 1:w + 1] = lr_copy
        # last row
        new_image[l - 1:l:1, 1:b - 1] = fr_copy
        # last column
        new_image[1:l - 1, b - 1:b] = fc_copy
        # first column
        new_image[1:l - 1, 0:1] = lc_copy
        new_image = np.array(new_image, dtype=np.uint8)
      
        pad_image=new_image.copy()
        gpyr=[pad_image]
        lpyr=[]
        nh, nw = new_image.shape[:2]
        for z in range(num_layers):
            if nh<4 or nw<4:
                break
            gauss_kern = np.array([[0.047459, 0.122933, 0.047459], [0.122933, 0.318432, 0.122933], [0.047459, 0.122933, 0.047459]])
            gauss_new_layer = np.zeros((h + 2, w + 2))
            for i in range(0, nh - 2):
                for j in range(0, nw - 2):
                    w1 = pad_image[i:i + 3, j:j + 3] * gauss_kern
                    gauss_new_layer[i, j] = np.sum(w1)
       
            gauss_new_layer = gauss_new_layer.astype("uint8")
            downsample_ip=np.copy(gauss_new_layer)
            dsi_h,dsi_w= downsample_ip.shape[:2]
            downsample_op=np.zeros((int(dsi_h/2),int(dsi_w/2)))
            dso_h,dso_w= downsample_op.shape[:2]
            downsample_op=downsample_ip[0::2,0::2]
            gpyr.append(downsample_op)
            print(downsample_op.shape)
            nh,nw=downsample_op.shape[:2]
            h=nh
            w=nw
            pad_image=pad_bw(downsample_op)

            print('downsized image : ',downsample_op.shape)
            scale_percent = 200 # percent of original size
            width = int(downsample_op.shape[1] * scale_percent / 100)
            height = int(downsample_op.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            resized = cv2.resize(downsample_op, dim, interpolation = cv2.INTER_NEAREST)
            print('Resized Dimensions : ',resized.shape)
            f,g=resized.shape[:2]
            gauss_new_layer=cv2.resize(gauss_new_layer,(g,f))
            print('gauss_new_layer:',gauss_new_layer.shape[:2])
            print('resized_image:',resized.shape[:2])
            laplacian_new = gauss_new_layer-resized
            laplacian_new = laplacian_new.astype("uint8")
            lpyr.append(laplacian_new)
            count.append(z)
            out =[gpyr, lpyr]

    else:
            #Padding the color image and performing processes on the color image
            
            h, w = ip_img_usr.shape[:2]
            fr_copy = ip_img_usr[0:1, ::1, ::]
            a = fr_copy.shape
            lr_copy = ip_img_usr[h - 1:h, ::1, ::]
            lc_copy = ip_img_usr[::1, w - 1:w, ::]
            fc_copy = ip_img_usr[::1, 0:1, ::]
            new_image = np.zeros((h_1 + 2, w_1 + 2, 3))
            pad_main = np.copy(ip_img_usr[0:h, 0:w, ::])
            l, b = new_image.shape[:2]
            new_image[1:h + 1, 1:w + 1, ::] = pad_main
            # first row
            new_image[0:1, 1:w + 1, ::] = lr_copy
            # last row
            new_image[l - 1:l:1, 1:b - 1, ::] = fr_copy
            new_image = np.array(new_image, dtype=np.uint8)
          
            nh, nw = new_image.shape[:2]
            print(new_image.shape)
            pad_image=new_image.copy()
            gpyr=[pad_image]
            lpyr=[]
        ###Our Gaussian Pyramid Begins here
            for z in range(num_layers):
                if nh<4 or nw<4:
                    break
                gauss_kern = np.array(
                        [[0.047459, 0.122933, 0.047459], [0.122933, 0.318432, 0.122933], [0.047459, 0.122933, 0.047459]])

                gauss_new_layer = np.zeros((h, w, 3))
                    #convolving with gaussian kernel
                for i in range(0, nh - 2):
                    for j in range(0, nw - 2):
                            b = pad_image[i:i + 3, j:j + 3, 0] * gauss_kern
                            gauss_new_layer[i, j, 0] = np.sum(b)
                            g = pad_image[i:i + 3, j:j + 3, 1] * gauss_kern
                            gauss_new_layer[i, j, 1] = np.sum(g)
                            r = pad_image[i:i + 3, j:j + 3, 2] * gauss_kern
                            gauss_new_layer[i, j, 2] = np.sum(r)
                    #The convolved image
                gauss_new_layer = gauss_new_layer.astype("uint8")
          
                downsample_ip=np.copy(gauss_new_layer)
                dsi_h,dsi_w= downsample_ip.shape[:2]
                downsample_op=np.zeros((int(dsi_h/2),int(dsi_w/2),3))
                dso_h,dso_w= downsample_op.shape[:2]
                downsample_op=downsample_ip[0::2,0::2,::]
                gpyr.append(downsample_op)
                print(downsample_op.shape)
                nh,nw=downsample_op.shape[:2]
                h=nh
                w=nw
                pad_image=pad_clr(downsample_op)
                    #gpyr.append(pad_image)       
                   
                print('downsized image : ',downsample_op.shape)
                scale_percent = 200 # percent of original size
                width = int(downsample_op.shape[1] * scale_percent / 100)
                height = int(downsample_op.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                resized = cv2.resize(downsample_op, dim, interpolation = cv2.INTER_NEAREST)
                print('Resized Dimensions : ',resized.shape)
              
                f,g=resized.shape[:2]
                downsample_ip=cv2.resize(downsample_ip,(g,f))
                print('gauss_new_layer:',gauss_new_layer.shape[:2])
                print('resized_image:',resized.shape[:2])
                laplacian_new = downsample_ip-resized  
                laplacian_new = laplacian_new.astype("uint8")
                lpyr.append(laplacian_new)
                count.append(z)
                out =[gpyr, lpyr]
    return out 
               

x=ComputePyr(ip_img_usr,num_layers)
print(type(x))
lpyr=x[1]
gpyr=x[0]
d=max(count)
gpyr.pop(d+1)
print((type(x)))
                
for i in range(len(lpyr)):
    cv2.imshow('lpyr'+str(i),lpyr[i])
    cv2.waitKey(800)
    
#for i in range(len(gpyr)):
    cv2.imshow('gpyr'+str(i),gpyr[i])
    cv2.waitKey(800)
    cv2.destroyAllWindows()

