####Script to analyze depth data and RGB data for orange coloured bricks
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Imageio:
    def __init__(self):
        """TODO: Docstring for function.
        :returns: TODO

        """
        self.rgb_image = None #RGB Image
        self.depth_image = None #Depth Image
        self.depth_image_gray = None #Depth Image in grayscale

    def load_image(self, rgb_img_path, depth_img_path):
        """Loads the RGB image and converts to grayscale. 

        """
        self.rgb_image = cv2.imread(rgb_img_path)
        self.display(self.rgb_image,"RGB Image")
        self.depth_image = cv2.imread(depth_img_path)
        self.depth_img_gray = cv2.cvtColor(self.depth_image,cv2.COLOR_BGR2GRAY)#Converting to grayscale

    
    def display(self,img,txt):
        """Member function to display the image 
        Parameters:
        img: Image to be displayed
        txt: Title

        """
        winName = txt
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName,img)
        key = cv2.waitKey(0)
        if( key & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            exit()

class analyzedepth(Imageio):

    """Class to analyze depth data for the orange brick
    using K-means clustering
    """

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()
        #self.rgb_image = None #RGB Image
        #self.depth_image = None #Depth Image    
        #self.depth_img_gray = None #Depth Image in grayscale
        self.cluster_result = None #Image which stores the result of KMeans clustering
        self.n_clusters = 2 #Number of clusters


    def cluster(self):
        """KMeans Clustering on the grayscale image
            Colour coded result in stored in self.cluster_result
        """
        
        #Preprocessing applied prior to Kmeans clustering
        #Applying CLAHE
        #Creating the CLAHE object
        #clahe = cv2.createCLAHE(clipLimit=40.0 , tileGridSize=(3,3))
        #cl1 = clahe.apply(self.depth_img_gray)
        self.depth_img_gray = cv2.equalizeHist(self.depth_img_gray)
        self.depth_img_gray = cv2.medianBlur(self.depth_img_gray,7)

        
        self.cluster_result = np.empty((self.depth_img_gray.shape[0]*self.depth_img_gray.shape[1],1))
        self.display(self.depth_img_gray,"Grayscale depth image")

        #Plotting the grayscale depth image 
        plt.title("Grayscale depth image")
        plt.imshow(self.depth_img_gray)
        plt.show()

        img_height = self.depth_img_gray.shape[0]
        img_width = self.depth_img_gray.shape[1]

        self.depth_img_gray = self.depth_img_gray.reshape((self.depth_img_gray.shape[0] * self.depth_img_gray.shape[1], 1)) #Reshaping the image

        print("Performing KMeans clustering with clusters=",self.n_clusters)
        #clt = MiniBatchKMeans(n_clusters,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
            
        #KMeans clustering using sklearns KMeans function
        clt = KMeans(self.n_clusters,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
        labels = clt.fit_predict(self.depth_img_gray)
       
        #Assigning a random colour to elements in cluster_result based on the value of labels
        for (i,label) in enumerate(labels):
            self.cluster_result[i] = (100*label)%255

        self.cluster_result = self.cluster_result.reshape((img_height,img_width))
        self.cluster_result = np.uint8(self.cluster_result) #Converting to uint8


        #Plotting the result of K Means clustering
        plt.title("K means clustering")
        plt.imshow(self.cluster_result)
        plt.show()

        self.display(self.cluster_result,"Result of clustering")

        #quant = clt.cluster_centers_.astype("uint8")[labels]
        #quant = clt.cluster_centers_.astype("uint8")

        #Reshaping
        #quant = quant.reshape((img_height,img_width,1))
        #self.depth_img_gray = self.depth_img_gray.reshape((img_height,img_width,1))


        #self.display(np.hstack([self.depth_img_gray,quant]),"K=2")


class analyzergb(Imageio):

    """Class to analyze the RGB Image"""

    def __init__(self):
        """TODO: to be defined. """
        super().__init__()
        self.img_yuv = None #YUV image
        self.v_channel = None #V channel
        self.mask = None #Mask image from histogram analysis
        self.edge_image = None #Edge Image obtained from Canny
        self.hough_image = None #Image on which hough lines are drawn

    def yuv_analyze(self):
        '''
        Function to analyze colour channel(V-channel of YUV colour scheme) using histogram analysis
        TODO: std_v to be dynamically calculated
        '''

        self.img_yuv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2YUV) #BGR2YUV conversion
        self.v_channel = self.img_yuv[:,:,2] #Extracting V channel
        self.display(self.v_channel,"V Channel")

        hist = cv2.calcHist([self.v_channel],[0],None,[256],[0,256])
        
        plt.plot(hist)
        plt.xlim([0,256])
        plt.show()


        
        std_v = 15.0   #TODO:This should be calculated dynamically
        
        #std_v = np.std(self.v_channel)
        #print("std_v:",std_v)
        mean = np.where(hist == np.amax(hist)) #Bin corresponding to the highest point in the histogram
        #print("mean:",mean)
        #lower_thresh_1 = 0
        #print("lower_thresh_1:",lower_thresh_1)
        #upper_thresh_1 = mean[0][0] - (std_v)
        #lower_thresh_2 = mean[0][0] + (std_v)
        #upper_thresh_2 = 255

        lower_thresh = mean[0][0] - (std_v)
        upper_thresh = mean[0][0] + (std_v)
        #print("lower_thresh:",lower_thresh)
        #print("upper_thresh:",upper_thresh)

        #Applying otsu thresholding
        #ret , thresh = cv2.threshold(self.v_channel , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print("Otsu threshold:",ret)

        self.mask = cv2.inRange(self.v_channel , lower_thresh , upper_thresh)

        self.display(self.mask, "Mask Image")          


    def test_image(self):
        '''Function to analyze different colourspaces and choose the best among them'''
        
        ### Canny Edge detection on the V channel in HSV colourspace ###
        test_image = np.copy(self.rgb_image)
        cnt_image = np.copy(test_image)
        img_hsv = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2HSV)
        #self.display(img_hsv,"HSV Image")
        channel = img_hsv[:,:,0]  #But here we are working on the intensity/brightness alone, not the colour component
        self.display(channel,"V Channel")

        #Plotting histogram of the V Channel

        hist = cv2.calcHist([channel],[0],None,[256],[0,256])
        
        plt.plot(hist)
        plt.xlim([0,256])
        plt.show()

        

        #Background Subtraction with the V channel in HSV colourspace 
        std_v = 33.0   #TODO:This should be calculated dynamically
        
        #std_v = np.std(self.v_channel)
        #print("std_v:",std_v)
        mean = np.where(hist == np.amax(hist)) #Bin corresponding to the highest point in the histogram
        #print("mean:",mean)
        #lower_thresh_1 = 0
        #print("lower_thresh_1:",lower_thresh_1)
        #upper_thresh_1 = mean[0][0] - (std_v)
        #lower_thresh_2 = mean[0][0] + (std_v)
        #upper_thresh_2 = 255

        lower_thresh = mean[0][0] - (std_v)
        upper_thresh = mean[0][0] + (std_v)
        #print("lower_thresh:",lower_thresh)
        #print("upper_thresh:",upper_thresh)

        #Applying otsu thresholding
        #ret , thresh = cv2.threshold(self.v_channel , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print("Otsu threshold:",ret)

        mask_image = cv2.inRange(channel , lower_thresh , upper_thresh)
        mask_image = cv2.bitwise_not(mask_image)

        #Opening to remove noise in the image
        kernel_opening = np.ones((19,19),np.uint8) #Determine the size of the kernel which is to be used
        mask_opening = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel_opening)

        #Dilation to fill holes in the foreground
        kernel_dilation = np.ones((5,5),np.uint8)
        mask_dilation = cv2.dilate(mask_opening,kernel_dilation,iterations = 1)

		#Display Mask images
        self.display(mask_image, "Mask Image")
        self.display(mask_opening,"Mask Image opening")
        self.display(mask_dilation,"Mask Dilation")

        #Obtaining contour from the mask Image
        cnt , hierarchy = cv2.findContours(mask_dilation , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

        #Finding index of the largest contour
        areas = [cv2.contourArea(c) for c in cnt]
        max_index = np.argmax(areas)
        largest_cnt = cnt[max_index]
        cv2.drawContours(cnt_image, [largest_cnt], 0, (0,255,0), 3)
        self.display(cnt_image, "Largest contour")

		####-----------------------------------------------------------------------------------------------------####
		#V channel cannot be used as it depends solely on intensity/brightness
		#Background subtraction does not work with Hue channel in HSV

        #Applying BilateralFiltering to remove noise and preserve edge information
        blur_img = cv2.bilateralFilter(channel,7,80,80) #TODO: Parameters for bilateralFilter function
        self.display(blur_img,"Blurred V Channel")

        #Low and high thresholds for canny edge detection
        sigma = 0.33
        mean = np.mean(blur_img)
        low_threshold = int(max(0,(1.0-sigma)*mean))
        high_threshold = int(min(255,(1.0+sigma)*mean))
        print("low threshold:",low_threshold)
        print("high threshold:",high_threshold)
        test_image = cv2.Canny(blur_img,low_threshold,high_threshold)
        self.display(test_image,"Edge Image")

            

    def find_edge(self):
        '''Function to find edge image using canny
        Stores the result in self.edge_image'''



        ###Background Subtracting on the V channel in HSV colourspace ###


        img_gray = cv2.cvtColor(self.rgb_image,cv2.COLOR_BGR2GRAY)#Converting to grayscale
        self.hough_image = np.copy(self.rgb_image)
        self.edge_image = np.zeros_like(img_gray)

        blur_img = cv2.bilateralFilter(img_gray,7,50,10) #Bilateral Filtering: A higher value is preferred for sigmacolour and a lower value for sigma space
        sigma = 0.33
        mean = np.mean(blur_img)
        low_threshold = int(max(0,(1.0-sigma)*mean))
        high_threshold = int(min(255,(1.0+sigma)*mean))

        low_threshold = 50
        high_threshold = 100
        print("low_threshold:",low_threshold)
        print("high_threshold:",high_threshold)
        self.edge_image = cv2.Canny(blur_img,low_threshold,high_threshold)

        self.display(self.edge_image,"Edge Image")

        #lines = cv2.HoughLines(self.edge_image,100,np.pi/90,1)

        #for rho,theta in lines[0]:
        #    a = np.cos(theta)
        #    b = np.sin(theta)
        #    x0 = a*rho
        #    y0 = b*rho
        #    x1 = int(x0 + 1000*(-b))
        #    y1 = int(y0 + 1000*(a))
        #    x2 = int(x0 - 1000*(-b))
        #    y2 = int(y0 - 1000*(a))

        #    cv2.line(self.hough_image,(x1,y1),(x2,y2),(0,0,255),2)

        #self.display(self.hough_image,"Hough Line Image")


if __name__ == '__main__':
    #Images to be analyzed(Orange Brick): file:///home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg , file:///home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg 
    #(Blue Brick): file:///home/varghese/brick_data/Dec_6/Dec_6_4/rgb_image_721.jpg , file:///home/varghese/brick_data/Dec_6/Dec_6_4/depth_image_721.jpg 
    #(Green Brick): file:///home/varghese/brick_data/Dec_6/Dec_6_5/rgb_image_137.jpg , file:///home/varghese/brick_data/Dec_6/Dec_6_5/depth_image_137.jpg 

    ###Analyze the depth Image 
    depth_obj = analyzedepth()
    
    #Orange Brick
    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg") #Loading the rgb and depth Images

    #Orange Brick
    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_rgb_image_787.jpg","/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_depth_image_787.jpg")

    #Blue Brick
    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_rgb_image_721.jpg", "/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_depth_image_721.jpg")

    #Green Brick
    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_rgb_image_137.jpg", "/home/varghese/brick_data/Dec_6/cropped_brick_data/cropped_depth_image_137.jpg")

    #Red Brick
    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/cropped_brick_data/rgb_image_377.jpg","/home/varghese/brick_data/Dec_6/cropped_brick_data/depth_image_377.jpg")
    
    #depth_obj.cluster()
    
    ###Analyze RGB Image
    rgb_obj = analyzergb()

    #Orange Brick
    #rgb_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg") #Loading the rgb and depth Images

    #Red Brick
    #rgb_obj.load_image("/home/varghese/brick_data/Dec_6/cropped_brick_data/rgb_image_377.jpg","/home/varghese/brick_data/Dec_6/cropped_brick_data/depth_image_377.jpg")

    #Blue Brick
    #rgb_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_4/rgb_image_721.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_4/depth_image_721.jpg") 

    #Green Brick
    rgb_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_5/rgb_image_137.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_5/depth_image_137.jpg")
    #rgb_obj.yuv_analyze()
    #rgb_obj.find_edge()
    rgb_obj.test_image()
