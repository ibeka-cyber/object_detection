import cv2 
import  numpy as np         
import time

np.random.seed(20)
class Detector:
    def __init__(self,videoPath,configPath,modelPath,classesPath) :
        self.videoPath =videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        ########################

        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

    def readClasses(self):
        with open (self.classesPath,"r") as f:
            self.classesList =f.read().splitlines()
        self.classesList.insert(0,'__Background__')
        self.colorList =  np.random.uniform(low=0,high=255,size=(len(self.classesList),3))
        #print(self.classesList) 

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Dosya açılırken hata oluştu")
            
        
        (success,image) = cap.read()
        
        while success:
            image = cv2.flip(image,1)
            classLabelIDs,confidences,bboxs = self.net.detect(image,confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))

            bboxsIdx = cv2.dnn.NMSBoxes(bboxs, confidences ,    score_threshold = 0.5,nms_threshold=0.2)

            if len(bboxs) !=0:
                for i in range(0,len(bboxsIdx)):
                    bbox = bboxs[np.squeeze(bboxsIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxsIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxsIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    
                    if classLabel == "person":
                        displayText ="{}".format("Insan:)")
                        
                    elif classLabel == "cup":
                        displayText ="{}".format("Bardak")
                    

                    
                    x, y, w, h = bbox
                    
                    if classLabel == "person" or classLabel == "cup":
                    
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=classColor, thickness=2)
                        cv2.putText(image,displayText,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,classColor,2)
    
                        #####################################
    
                        lineWidth = min(int(w*0.3),int(h*0.3))
                        cv2.line(image,(x,y),(x+lineWidth,y),color=classColor,thickness=10)
                        cv2.line(image,(x,y),(x,y+lineWidth),color=classColor,thickness=10)
                        
                        cv2.line(image,(x+w,y),(x+w-lineWidth,y),color=classColor,thickness=10)
                        cv2.line(image,(x+w,y),(x+w ,y+lineWidth),color=classColor,thickness=10)
    
                        cv2.line(image,(x,y+h),(x+lineWidth,y+h),color=classColor,thickness=10)
                        cv2.line(image,(x,y+h),(x,y+h-lineWidth),color=classColor,thickness=10)
                        
                        cv2.line(image,(x+w,y+h),(x+w-lineWidth,y+h),color=classColor,thickness=10)
                        cv2.line(image,(x+w,y+h),(x+w ,y+h-lineWidth),color=classColor,thickness=10)
            
            
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result",image)
            
            key = cv2.waitKey(1) & 0xFF
            if key ==ord("q"):
                break
            (success,image) = cap.read()
        cv2.destroyAllWindows()