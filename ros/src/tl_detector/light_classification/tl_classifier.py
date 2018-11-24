from styx_msgs.msg import TrafficLight
import cv2
import os
import keras
import numpy as np
from keras.applications import vgg16
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #pass
        
        self.threshold = .8
        
        #Load the VGG16 model
        #vgg_model = vgg16.VGG16(weights='imagenet')
        #os.chdir('.')
        #model = load_model('light_classification/highway_modelv2-ep20.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        return TrafficLight.UNKNOWN

"""   
        #TODO implement light color prediction
        image224 = cv2.resize(image, (224, 224))
        image224 = img_to_array(image224)
        processed_image_vgg16 = vgg16.preprocess_input(image224.copy())
        predictions_vgg16 = vgg_model.predict(processed_image_vgg16)
        label_vgg16 = decode_predictions(predictions_vgg16)


     
        if (int(label_vgg16[0][0][1] == 'traffic_light') & int(label_vgg16[0][0][2] > 0.7)):
            # make a prediction
            predict = model.predict(np.expand_dims(image224, axis=0))
            
            if predict[0] > self.threshold:
                print('RED')
                return TrafficLight.RED
            elif predict[1] > self.threshold:
                print('YELLOW')
                return TrafficLight.YELLOW
            elif predict[2] > self.threshold:
                print('GREEN')
                return TrafficLight.GREEN
           return TrafficLight.UNKNOWN
"""