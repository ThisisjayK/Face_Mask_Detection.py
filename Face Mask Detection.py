# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from smbus2 import SMBus
from mlx90614 import MLX90614
# from RPi_GPIO_i2c_LCD import lcd
from rpi_lcd import LCD
import Adafruit_DHT
import RPi.GPIO as GPIO
import time
import os,sys
import spidev # To communicate with SPI devices
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
#***************************** Defining all use pin ***************************************#  
## Address of backpack
# i2c_address = 0x27

## Initalize display
# LCD = lcd.HD44780(i2c_address)
LCD1 = LCD()


#Define temp sensor DHT22
# DHT_SENSOR = Adafruit_DHT.DHT22
# DHT_PIN = 4

I2C_BUS = 1
SENSOR_ADDRESS = 0x5A
# Initialize the I2C bus
bus = SMBus(I2C_BUS)

# Initialize the MLX90614 sensor
sensor = MLX90614(bus, address=SENSOR_ADDRESS)

green_led_pin = 29
Red_led_pin = 33
servo_pin=31
buzzer_pin = 36


GPIO.setup(servo_pin,GPIO.OUT)
pwm = GPIO.PWM(servo_pin,50) # 50 Hz (20 ms PWM period)
pwm.start(7) # start PWM by rotating to 90 degrees
pwm.ChangeDutyCycle(0)
 
 
GPIO.setup(Red_led_pin , GPIO.OUT)
GPIO.setup(green_led_pin , GPIO.OUT) 
GPIO.setup(buzzer_pin, GPIO.OUT)
s=1
flag='c'
#***************************** Read Temperature code start ***************************************#    
# Start SPI connection
def Temp():
    temperature = sensor.get_object_1()
#     humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
    return temperature
#***************************** Read Temperature code end ***************************************#


#***************************** detect function ***************************************#
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
#***************************** detect function end ***************************************#

# load our serialized face detector model from disk
prototxtPath = r"/home/facemask/Downloads/deploy.prototxt"
weightsPath = r"/home/facemask/Downloads/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("/home/facemask/Downloads/mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(0).start()

#***************************** software function ***************************************#
def software():
    s=1
    flag='c'

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # include the probability in the label
            label1 = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label1, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # show the output frame
            cv2.imshow("Frame", frame)
            cv2.waitKey(1000)
            if(label == "Mask"):
                print("mask detected")
                flag='a'
                s=0
            else:
                print("mask not detected")
                flag='b'
                s=0

        if s == 0:
            break
    return flag
#***************************** software function end ***************************************#



#***************************** Main code ***************************************# 
LCD1.text("WELCOME TO FACE",1)
LCD1.text("MASK DETECTION",2)
time.sleep(5) 
while(1):
    LCD1.clear()
    
    LCD1.text("Please Scan",1)
    LCD1.text("Your Temperature",2)
    time.sleep(5)
    temp  = Temp()
    itemp=int(temp)
    LCD1.clear()
    LCD1.text("Your Body Temp: ",1)
    LCD1.text(str(itemp),2)
    time.sleep(5)
    print(temp)
            
    LCD1.clear()
    LCD1.text("Please Scan ",1)
    LCD1.text("Your Face ",2)
    time.sleep(1)
    if(temp<35):
        flag=software()
        if((flag == 'a') ):
            LCD1.clear()
            LCD1.text("Gate Open",1)
            time.sleep(2)
            GPIO.output(green_led_pin,GPIO.HIGH)  #LED ON
            #GPIO.output(Red_led_pin,GPIO.LOW)  #LED OFF
            #pwm.start(7)
            #pwm.ChangeDutyCycle(0)
            pwm.ChangeDutyCycle(2.0) # rotate to 0 degrees
            time.sleep(1)  
            pwm.ChangeDutyCycle(7.0) # rotate to 90 degrees
            time.sleep(1)
            pwm.ChangeDutyCycle(0)
            # stops the pwm on 13
            #pwm.stop()
                    
            GPIO.output(green_led_pin,GPIO.LOW)
                    
        elif(flag == 'b'):
            LCD1.clear()
            LCD1.text("Please wear mask",1)
            # LCD.clear()
            #GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
            GPIO.output(Red_led_pin,GPIO.HIGH)  #LED ON
            GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer On
            time.sleep(2)
            GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF
            GPIO.output(Red_led_pin,GPIO.LOW)
    else:
        LCD1.clear()
        LCD1.text("High Body Temperature ",1)
        #GPIO.output(green_led_pin,GPIO.LOW)  #LED OFF
        GPIO.output(Red_led_pin,GPIO.HIGH)  #LED ON
        GPIO.output(buzzer_pin,GPIO.HIGH)  #Buzzer On
        time.sleep(0.5)
                
        GPIO.output(buzzer_pin,GPIO.LOW)  #Buzzer OFF
        GPIO.output(Red_led_pin,GPIO.LOW)
            #cv2.putText(img,n,(x,y),font,1,(255,100,100),2) #show image
        #cv2.imshow("output",img) #show image
        # 27 is ASCII value of escapse key
        #It is use to run thread which waiting for ESC button becuase of this image will show on window
        #if you not use this function then it will not load image
    if((cv2.waitKey(2) == 27)):
            LCD1.clear()
            pwm.stop()
            break;
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


