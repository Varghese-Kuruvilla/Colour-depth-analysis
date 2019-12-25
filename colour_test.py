#Script to obtain the step from a top down view of the construction zone
import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
import matplotlib.pyplot as plt


def breakpoint():
    inp = input("Enter any input...")
    if(inp == 'q'):
        cv2.destroyAllWindows()
        exit()

def edge_detect(img):
    display(img,"RGB Image")

    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    display(img_gray,"Grayscale Image")

    
    bil_img = cv2.bilateralFilter(img_gray,9,80,80)
    display(bil_img, "Filtered Image")
    

    #lower_thresh = 120
    #upper_thresh = 150

    #v = np.median(bil_img)
    #sigma = 0.33
    #lower_thresh = int(max(0,(1.0-sigma)*v))
    #upper_thresh = int(min(255,(1.0+sigma)*v))
    #print("Lower_thresh:",lower_thresh)
    #print("upper_thresh:",upper_thresh)

    ret , thresh = cv2.threshold(bil_img , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Otsu threshold:",ret)
    upper_thresh = ret
    lower_thresh = 0.5 * ret
    print("Lower_thresh:",lower_thresh)
    print("upper_thresh:",upper_thresh)
    edge_img = cv2.Canny(bil_img,lower_thresh,upper_thresh,apertureSize=3)
    display(edge_img,"Edge image")




    #Hough Lines
    rho_ls = []
    theta_ls = []
    line_img = np.copy(img)
    lines = cv2.HoughLines(edge_img,1,np.pi/180,100)
    lines_small = cv2.HoughLines(edge_img,1,np.pi/180,5)
    print("lines:",lines)
    print("type(lines)",type(lines))
    print("lines.shape:",lines.shape)
    for element in lines:
        rho = element[0][0]
        theta = element[0][1]
        #Try to find lines that are perpendicular to this
        for (i,element_1) in enumerate(lines_small):
            rho_1 = element_1[0][0]
            theta_1 = element_1[0][1]
            if (((theta_1 - theta) <= 1.58) and ((theta_1 - theta) >= 1.55)):
                #print("theta_1:",theta_1)
                #print("theta:",theta)
                if(i==0):
                    rho_ls.append(rho)
                    theta_ls.append(theta)
                
                rho_ls.append(rho_1)
                theta_ls.append(theta_1)


    #print("rho_ls:",rho_ls)
    #print("theta_ls:",theta_ls)

    for i in range(0,len(rho_ls)):
        rho = rho_ls[i]
        theta = theta_ls[i]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if(i <= 400):
            cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)

    display(line_img,"Result of Hough Lines")

    print("len(rho_ls):",len(rho_ls))
    print("len(theta_ls):",len(theta_ls))
    
    #cnt_img = np.copy(img)
    #contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #for cnt in contours:
    #    cv2.drawContours(cnt_img,cnt, -1 , (0,255,0), 3)
    #    display(cnt_img,"Contour Image")

def color_analyze(img):
    '''In the grayscale image the edges are not pronounced
    Edges are not pronounced in the Hue channel
    Saturation channel is giving decent edges(needs to be tested more)
    To find the white sheet we could use the Saturation channel from HSV space'''


    display(img,"RGB Image")
    #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #display(img_gray,"GrayScale Image")

    #HSV colourspace
    img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #display(img_hsv,"HSV image")
    

    #img_hue = img_hsv[:,:,0]
    #display(img_hue, "Hue Channel")

    img_sat = img_hsv[:,:,1]
    display(img_sat, "Sat Channel")


    #img_sat_histeq = cv2.equalizeHist(img_sat)
    #display(img_sat_histeq,"Saturation channel with histogram Equalization")

    #img_value = img_hsv[:,:,2]
    #display(img_value,"Value Channel")
    


    #YUV colourspace
    #img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    #display(img_yuv,"YUV Image")

    #img_y = img_yuv[:,:,0]
    #display(img_y,"Y channel of YUV")

    #img_u = img_yuv[:,:,1]
    #display(img_u,"U channel of YUV")

    #img_v = img_yuv[:,:,2]
    #display(img_v," V channel of YUV")


    #LAB Colourspace
    #img_lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    #display(img_lab,"Lab image")

    #img_l = img_lab[:,:,0]
    #display(img_l,"L channel of LAB")

    #img_a = img_lab[:,:,1]
    #display(img_a , "A channel of LAB")

    #img_b = img_lab[:,:,2]
    #display(img_b," B channel of LAB")
    bil_img = cv2.bilateralFilter(img_sat,3,80,80)
    display(bil_img, "Filtered Image")

    #Plotting histogram of the filtered image
    hist = cv2.calcHist([img_sat],[0],None,[256],[0,256])
    
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()


    #Background subtraction
    channel = np.copy(img_sat)
    mean = np.where(hist == np.amax(hist))
    std_v = np.std(channel)
    lower_thresh = mean[0][0] - std_v
    upper_thresh = mean[0][0] + std_v
    print("Mean:",mean)
    print("std_v:",std_v)
    mask = cv2.inRange(channel,lower_thresh,upper_thresh)
    mask_inv = cv2.bitwise_not(mask)
    display(mask_inv,"Mask Image")

    #Opening- For removing external noise
    #kernel = np.ones((17,17),np.uint8)
    #opening_img = cv2.morphologyEx(mask_inv,cv2.MORPH_OPEN,kernel)
    #display(opening_img,"Result of Opening")

    #Finding contours
    cnt_img = np.copy(img)
    contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("hierarchy:",hierarchy)
    cv2.drawContours(cnt_img,contours, -1 , (0,255,0), 3)
    display(cnt_img,"Contour Image")

    #Drawing the largest contour on the image
    img_max_area = np.copy(img)
    areas = [cv2.contourArea(c) for c in contours]
    max_area = np.argmax(areas)
    cv2.drawContours(img_max_area,contours[max_area],-1,(0,255,0),3)
    display(img_max_area,"img_max_area")

    #for cnt in contours:
    #    cnt_img = cv2.drawContours(cnt_img,cnt, -1 , (0,255,0) , 3)
    #    display(cnt_img, "Contour Image")
    #Thresholds for canny-statistics
    #v = np.median(bil_img)
    #sigma = 0.33
    #lower_thresh = int(max(0,(1.0-sigma)*v))
    #upper_thresh = int(min(255,(1.0+sigma)*v))
    #edge_img = cv2.Canny(bil_img,lower_thresh,upper_thresh)
    #display(edge_img,"Edge image")

    #Thresholds for canny-Otsu
    #ret , thresh = cv2.threshold(bil_img , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("Otsu threshold:",ret)
    #upper_thresh = ret
    #lower_thresh = 0.5 * ret
    #edge_img = cv2.Canny(bil_img,lower_thresh,upper_thresh)
    #display(edge_img,"Edge image")

    ####-----------------------------------------------------------------------------------------------####


def display(img,txt):
    winName = txt
    cv2.namedWindow(winName,cv2.WINDOW_NORMAL)
    cv2.imshow(winName,img)
    key = cv2.waitKey(0)
    if(key & 0xFF == ord('q')):
        cv2.destroyAllWindows()
        exit()


if __name__ == '__main__':
    
    #rgb_img = cv2.imread("/home/varghese/brick_data/data_dec_23/cropped_images/rgb_image_24.jpg")
    rgb_img = cv2.imread("/home/varghese/brick_data/data_dec_23/cropped_images/rgb_image_24.jpg")
    #color_analyze(rgb_img)    
    edge_detect(rgb_img)
