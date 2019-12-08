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
        self.n_clusters = 5 #Number of clusters


    def cluster(self):
        """KMeans Clustering on the grayscale image
            Colour coded result in stored in self.cluster_result
        """

        self.cluster_result = np.empty((self.depth_img_gray.shape[0]*self.depth_img_gray.shape[1],1))

        #Plotting the grayscale depth image 
        plt.title("Grayscale depth image")
        plt.imshow(self.depth_img_gray)
        plt.show()

        img_height = self.depth_img_gray.shape[0]
        img_width = self.depth_img_gray.shape[1]

        self.depth_img_gray = self.depth_img_gray.reshape((self.depth_img_gray.shape[0] * self.depth_img_gray.shape[1], 1)) #Reshaping the image

        print("Performing KMeans clustering with clusters=",self.n_clusters)
        #clt = MiniBatchKMeans(n_clusters,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
            
        #KMeans using sklearns KMeans function
        clt = KMeans(self.n_clusters,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
        labels = clt.fit_predict(self.depth_img_gray)
       
        #Assigning a random colour to elements in cluster_result based on the value of labels
        for (i,label) in enumerate(labels):
            self.cluster_result[i] = (100*label)%255

        self.cluster_result = self.cluster_result.reshape((img_height,img_width))

        #Plotting the result of K Means clustering
        plt.title("K means clustering")
        plt.imshow(self.cluster_result)
        plt.show()



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

    def color_analyze(self):
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
    
    

if __name__ == '__main__':
    #Images to be analyzed(Orange Brick): file:///home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg , file:///home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg 


    ###Analyze the depth Image 
    #depth_obj = analyzedepth()

    #depth_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg") #Loading the rgb and depth Images

    #depth_obj.cluster()
    
    ###Analyze RGB Image
    rgb_obj = analyzergb()
    rgb_obj.load_image("/home/varghese/brick_data/Dec_6/Dec_6_3/rgb_image_787.jpg", "/home/varghese/brick_data/Dec_6/Dec_6_3/depth_image_787.jpg") #Loading the rgb and depth Images
    rgb_obj.color_analyze()
