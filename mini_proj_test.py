import os
import shutil
import csv
import numpy as np
import dlib
import cv2
import timeit
from skimage.measure import compare_ssim as ssim
from keras.preprocessing import image
import matplotlib.pyplot as plt

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = 'C:\\Users\\TheoV\\PycharmProjects\\untitled\\venv\\dataset'
images_dir = os.path.join(basedir,'celeba')
labels_filename = os.path.join(basedir,'attribute_list.csv')

image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
target_size = None

def face_detect_gffd(saveimg):
    print("starting gffd detection")
    start = timeit.default_timer()
    outliers_pred = np.array([])
    facedetector = dlib.get_frontal_face_detector()
    for imgidx in range(len(saveimg)) :
        facedetected = facedetector(saveimg[imgidx],1)
        if len(facedetected) == 0 :
            outliers_pred = np.append(outliers_pred, imgidx+1)
            # imgfile = str(imgidx + 1) + ".png"
            # shutil.copy(os.path.join(images_dir, imgfile), os.path.join(basedir, "outliers"))     #used to copy all outliers to another folder
    stop = timeit.default_timer()
    print('Time gffd: ', stop - start)
    return outliers_pred

def face_detect_ssim(saveimg,meanimg, thresh = 0.4):
    print("starting ssim detection")
    start = timeit.default_timer()
    ssimarr = np.zeros([len(saveimg)])
    outliers_pred = np.array([])
    for imgidx in range(len(saveimg)) :
        ssimarr[imgidx] = ssim(meanimg, saveimg[imgidx], multichannel=False)
        if ssimarr[imgidx] < thresh :
            outliers_pred = np.append(outliers_pred, imgidx + 1)
            # imgfile = str(imgidx+1)+".png"
            # shutil.copy(os.path.join(images_dir, imgfile), os.path.join(basedir, "outliers"))
    # plt.bar(np.arange(1,len(saveimg)+1),ssimarr)
    # plt.show()
    stop = timeit.default_timer()
    print('Time ssim: ', stop - start)
    return outliers_pred

def face_detect_haar(saveimg) :
    print("starting haar detection")
    start = timeit.default_timer()
    face_cascade = cv2.CascadeClassifier('C:\\Users\\TheoV\\PycharmProjects\\untitled\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    outliers_pred = np.array([])
    for imgidx in range(len(saveimg)):
        facedetected = face_cascade.detectMultiScale(saveimg[imgidx], 1.3, 5)
        if type(facedetected)is tuple :
            outliers_pred = np.append(outliers_pred, imgidx+1)
    stop = timeit.default_timer()
    print('Time haar: ', stop - start)
    return outliers_pred

def get_real_outliers(imglabels) :
    outliers_real = np.array([])
    for imgidx in range(len(imglabels)):
        if np.array_equal(imglabels[imgidx, 0:],[-1, -1, -1, -1, -1]):
            outliers_real = np.append(outliers_real, imgidx + 1)
    return outliers_real

def get_mean_img(saveimg):
# get a mean image by taking the mean value of each pixel for all the pictures of non human faces and outliers in the database
    meanimg = np.zeros([256,256])
    for imgidx in range(len(saveimg)) :
        meanimg = np.add(meanimg,saveimg[imgidx-1])
    meanimg = (np.divide(meanimg,len(saveimg))).astype(np.uint8)
    # plt.imshow(meanimg)
    # plt.show()
    return meanimg

def accuracy_comp(real,pred,labels) :
    truenegarray = [x for x in real if x in pred]
    trueposarray = [x for x in np.arange(1,len(labels)+1) if x not in real and x not in pred]
    falsenegarray = [x for x in pred if x not in real]
    falseposarray = [x for x in real if x not in pred]
    print("true negatives = ", len(truenegarray))
    print("false positives = ", len(falseposarray))
    print("false negatives = ", len(falsenegarray))
    print("true positives = ", len(trueposarray))
    accur = (len(truenegarray)+len(trueposarray))/(len(labels))*100
    print("accuracy = ", accur, "%")
    print("\n")

#Open all images and save values
if os.path.isfile(os.path.join(basedir,"saveimg.npy")) :
    saveimg = np.load(os.path.join(basedir,"saveimg.npy"))
    saveimgcolor = np.load(os.path.join(basedir, "saveimgcolor.npy"))
    print("image array loaded")
elif os.path.isdir(images_dir) :
    saveimg = np.empty([5000, 256, 256], dtype=np.uint8)
    saveimgcolor = np.empty([5000, 256, 256, 3], dtype=np.uint8)
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                           target_size=target_size,
                           interpolation='bicubic'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        saveimg[int(file_name)-1] =  gray     #save gray img pixel values to array for later processing
        saveimgcolor[int(file_name)-1] =  img
    saveimg = saveimg.astype(np.uint8)  #convert img to uint8, useful for imshow
    np.save(os.path.join(basedir,"saveimg"),saveimg)
    saveimg = saveimgcolor.astype(np.uint8)  # convert img to uint8, useful for imshow
    np.save(os.path.join(basedir, "saveimgcolor"), saveimgcolor)
    print("image array storing done")
else :
    print("image folder not found")

#open csv file to store labels or load ndarray with labels
if os.path.isfile(os.path.join(basedir,"imglabels.npy")) :
    imglabels = np.load(os.path.join(basedir,"imglabels.npy"))
    print("labels loaded")
    for row in imglabels :
        if row[len(row) - 1] == "-1":
            humanbinar = np.append(humanbinar, float(row[0]))
else :
    with open(labels_filename,'r') as csvfile:
        labelscsv = csv.reader(csvfile, delimiter=',')
        humanbinar = np.array([])
        imglabels = np.zeros([5000, 5])
        counter = 1;
        for row in labelscsv:
            if counter >= 3 : # ignore headers for csv file
                imglabels[int(row[0])-1,:] = (row[1:])
            if row[len(row)-1] == "-1":
                humanbinar = np.append(humanbinar, float(row[0]))
            counter += 1
        imglabels.astype(int)
        np.save(os.path.join(basedir,"imglabels.npy"),imglabels)
        print("labels saved")

if not os.path.isdir(os.path.join(basedir, "outliers")):
    os.mkdir(os.path.join(basedir, "outliers"))

meanimg = get_mean_img(saveimg)
outliers_real = np.sort(get_real_outliers(imglabels))
outliers_pred_haar = np.sort(face_detect_haar(saveimg))
outliers_pred_frontal = np.sort(face_detect_gffd(saveimg))
outliers_pred_ssim = np.sort(face_detect_ssim(saveimg,meanimg))

print("\n gffd accuracy:")
accuracy_comp(outliers_real,outliers_pred_frontal,np.arange(1,len(imglabels)+1))
print("\n ssim accuracy:")
accuracy_comp(outliers_real,outliers_pred_ssim,np.arange(1,len(imglabels)+1))
print("\n haar accuracy:")
accuracy_comp(outliers_real,outliers_pred_haar,np.arange(1,len(imglabels)+1))





