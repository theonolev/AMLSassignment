import os
import shutil
import csv
import numpy as np
import dlib
import cv2
import timeit
from random import randint
from skimage.measure import compare_ssim as ssim
from keras.preprocessing import image
from sklearn import svm,metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

# PATH TO ALL IMAGES
global basedir, image_paths, target_size, img_size, starttime
basedir = 'C:\\Users\\TheoV\\PycharmProjects\\untitled\\venv\\dataset'
images_dir = os.path.join(basedir,'celeba')
labels_filename = os.path.join(basedir,'attribute_list.csv')
img_size = 64

image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
target_size = None
starttime = timeit.default_timer()

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
            # shutil.copy(os.path.join(images_dir, imgfile),os.path.join(basedir, "outliers"))  # used to copy all outliers to another folder
            # shutil.move(os.path.join(images_dir, imgfile), os.path.join(os.path.join(basedir, "outliers"), imgfile))     #used to move all outliers to another folder
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
        if np.array_equal(imglabels[imgidx, 1:],[-1, -1, -1, -1, -1]):
            outliers_real = np.append(outliers_real, imgidx + 1)
    return outliers_real


def get_mean_img(saveimg):
# get a mean image by taking the mean value of each pixel for all the pictures of non human faces and outliers in the database
    meanimg = np.zeros([256,256])
    for imgidx in range(len(saveimg)) :
        meanimg = np.add(meanimg,saveimg[imgidx-1])
    meanimg = (np.divide(meanimg,len(saveimg))).astype(np.uint8)
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


def get_datasets(saveimg, imglabels, trainval = 0.8, testval = 0.2, valid = False):
    shuffledidx = np.arange(len(imglabels))
    np.random.shuffle(shuffledidx)

    if valid is True :
        validval = trainval*0.2
        trainval -= validval
        validlen = int(len(saveimg)*validval)
        trainlen = int(len(saveimg) * trainval)
        testlen = int(len(saveimg) * testval)
        X_train = saveimg[shuffledidx[:trainlen]]
        X_valid = saveimg[shuffledidx[np.arange(trainlen,trainlen+validlen)]]
        X_test = saveimg[shuffledidx[trainlen+validlen:]]
        Y_train = imglabels[shuffledidx[:trainlen]]
        Y_valid = imglabels[shuffledidx[np.arange(trainlen, trainlen + validlen)]]
        Y_test = imglabels[shuffledidx[trainlen + validlen:]]
        return X_train, X_test, X_valid, Y_train, Y_test, Y_valid
    else :
        trainlen = int(len(saveimg) * trainval)
        testlen = int(len(saveimg) * testval)
        X_train = saveimg[shuffledidx[:trainlen]]
        X_test = saveimg[shuffledidx[trainlen:]]
        Y_train = imglabels[shuffledidx[:trainlen]]
        Y_test = imglabels[shuffledidx[trainlen:]]
        return X_train, X_test, Y_train, Y_test


def createModel(multiclass = False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size,img_size,3)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    if multiclass :
        model.add(Dense(7, activation='softmax'))
    else:
        model.add(Dense(3, activation='softmax'))

    return model


def SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx) :
    print("\nstarting task %i with SVM" % (filetaskidx))
    print("creating SVM")
    clf = svm.SVC(kernel="poly", gamma='scale', degree=3)
    print("training SVM")
    clf.fit(X_train, Y_train[:, taskidx])
    print("making predictions")
    predic = clf.predict(X_test)
    accur = metrics.balanced_accuracy_score(Y_test[:, taskidx], predic)
    predic_kfold = cross_val_predict(clf, X_test, Y_test[:, taskidx], cv=3)
    scores = cross_val_score(clf, X_test, Y_test[:, taskidx], cv=3)
    confus = metrics.confusion_matrix(Y_test[:, taskidx], predic)
    print("accuracy before cross validation = ", accur)  # 31/12/2018 => accuracy = 0.81
    print("confusion matrix : \n", confus)
    print("mean accuracy after cross validation = ", scores.mean())
    print("test accuracy after cross validation = ", metrics.balanced_accuracy_score(Y_test[:, taskidx], predic_kfold))
    stoptime = timeit.default_timer()
    print("exec time = ", stoptime - starttime)
    csvtitle = "task_%i.csv" % (filetaskidx)
    with open(os.path.join(basedir, csvtitle), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["average inference accuracy = %f" % (accur)])
        for idx in range(len(Y_test)):
            row = zip(["%i.png" % (Y_test[idx, 0].astype(int))], [predic[idx].astype(int)])
            wr.writerows(row)


def CNN_solve_task(X_train_CNN, X_test_CNN, X_valid_CNN,Y_train_CNN, Y_test_CNN, Y_valid_CNN,taskidx, filetaskidx) :
    # taskidx = 1
    print("CNN model creation")
    if filetaskidx != 5:
        model1 = createModel()
    else :
        model1 = createModel(multiclass = True)
    print("CNN model done")
    batch_size = 60
    epochs = 3
    model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    nb_classes = np.max(Y_train_CNN[:, taskidx].astype(int) + 2)
    if filetaskidx != 5 :
        Y_train_CNN_task = Y_train_CNN[:,taskidx].clip(min=0)
        Y_valid_CNN_task = Y_valid_CNN[:,taskidx].clip(min=0)
        Y_test_CNN_task = Y_test_CNN[:,taskidx].clip(min=0)
    else :
        Y_train_CNN_task = Y_train_CNN[:, taskidx] + 1
        Y_valid_CNN_task = Y_valid_CNN[:, taskidx] + 1
        Y_test_CNN_task = Y_test_CNN[:, taskidx] + 1

    # normalize image colour values (from range 0-255 to range 0-1) so it doesn't kill weight in the activation layers
    X_train_CNN = X_train_CNN / 255
    X_test_CNN = X_test_CNN / 255
    X_valid_CNN = X_valid_CNN / 255

    Y_train_CNN_task = np_utils.to_categorical(Y_train_CNN_task, nb_classes)
    Y_valid_CNN_task = np_utils.to_categorical(Y_valid_CNN_task, nb_classes)
    Y_test_CNN_task = np_utils.to_categorical(Y_test_CNN_task, nb_classes)
    history = model1.fit(X_train_CNN, Y_train_CNN_task, batch_size=batch_size, epochs=epochs, verbose=1,
                         validation_data=(X_valid_CNN, Y_valid_CNN_task))
    print("\nCNN model evaluation")
    predic_CNN = model1.predict(X_test_CNN)  
    if filetaskidx != 5:
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN]) - 1
    else :
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN])
    mod_eval = model1.evaluate(X_test_CNN, Y_test_CNN_task)
    eval_accur = mod_eval[1]

    confus = metrics.confusion_matrix(Y_test_CNN[:, taskidx], predic_CNN)
    print("accuracy = ", eval_accur)
    print("confusion matrix : \n", confus)
    stoptime = timeit.default_timer()
    print("exec time = ", stoptime - starttime)

#=======================================================================================================================
#Open all images and save values
if os.path.isfile(os.path.join(basedir,"saveimg.npy")) :
    saveimg = np.load(os.path.join(basedir,"saveimg.npy"))
    saveimgcolor = np.load(os.path.join(basedir, "saveimgcolor.npy"))
    print("image array loaded")
elif os.path.isdir(images_dir) :
    saveimg = np.empty([5000, img_size, img_size], dtype=np.uint8)
    saveimgcolor = np.empty([5000, img_size, img_size, 3], dtype=np.uint8)
    for img_path in image_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                           target_size=target_size,
                           interpolation='bicubic'))
        img = cv2.resize(img, (img_size, img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        saveimg[int(file_name)-1] =  gray     #save gray img pixel values to array for later processing
        saveimgcolor[int(file_name)-1] =  img
    saveimg = saveimg.astype(np.uint8)  #convert img to uint8, useful for imshow
    np.save(os.path.join(basedir,"saveimg"),saveimg)
    saveimgcolor = saveimgcolor.astype(np.uint8)  # convert img to uint8, useful for imshow
    np.save(os.path.join(basedir, "saveimgcolor"), saveimgcolor)
    print("image array storing done")
else :
    print("image folder not found")

#open csv file to store labels or load ndarray with labels
if os.path.isfile(os.path.join(basedir,"imglabels.npy")) :
    imglabels = np.load(os.path.join(basedir,"imglabels.npy"))
    print("labels loaded")
    # for row in imglabels :
    #     if row[len(row) - 1] == "-1":
    #         humanbinar = np.append(humanbinar, float(row[0]))
else :
    with open(labels_filename,'r') as csvfile:
        labelscsv = csv.reader(csvfile, delimiter=',')
        # humanbinar = np.array([])
        imglabels = np.zeros([5000, 6])
        counter = 1;
        for row in labelscsv:
            if counter >= 3 : # ignore headers for csv file
                currlabel = np.append(int(row[0]),row[1:])
                imglabels[int(row[0])-1,:] = currlabel
            # if row[len(row)-1] == "-1":
            #     humanbinar = np.append(humanbinar, float(row[0]))
            counter += 1
        imglabels.astype(int)
        np.save(os.path.join(basedir,"imglabels.npy"),imglabels)
        print("labels saved")

if not os.path.isdir(os.path.join(basedir, "outliers")):
    os.mkdir(os.path.join(basedir, "outliers"))
#=======================================================================================================================

#=======================================================================================================================
#Task A: face detection to remove outliers
#tests with different methods for face detection: SSIM, HAAR cascade and HOG => HOG seems most accurate with acc = 95%

outliers_real = np.sort(get_real_outliers(imglabels))
if os.path.isfile(os.path.join(basedir,"outliers.npy")) :
    outliers_pred_frontal = np.load(os.path.join(basedir, "outliers.npy"))
else :
    outliers_pred_frontal = np.sort(face_detect_gffd(saveimg))
    np.save(os.path.join(basedir, "outliers"), outliers_pred_frontal)
saveimg_gffd = np.delete(saveimg, [x-1 for x in outliers_pred_frontal], axis=0)
saveimgcolor_gffd = np.delete(saveimgcolor, [x-1 for x in outliers_pred_frontal], axis=0)
imglabels_gffd = np.delete(imglabels, [x-1 for x in outliers_pred_frontal], axis=0)

# meanimg = get_mean_img(saveimg)
# outliers_pred_haar = np.sort(face_detect_haar(saveimg))
# outliers_pred_ssim = np.sort(face_detect_ssim(saveimg,meanimg))

print("\ngffd accuracy:")
accuracy_comp(outliers_real,outliers_pred_frontal,np.arange(1,len(imglabels)+1))
# print("\n ssim accuracy:")
# accuracy_comp(outliers_real,outliers_pred_ssim,np.arange(1,len(imglabels)+1))
# print("\n haar accuracy:")
# accuracy_comp(outliers_real,outliers_pred_haar,np.arange(1,len(imglabels)+1))
#=======================================================================================================================

#=======================================================================================================================
#Task B dataset split to determine training set, testing set and if needed validation set
X_train, X_test, Y_train, Y_test, = get_datasets(saveimg_gffd,imglabels_gffd)
X_train_CNN, X_test_CNN, X_valid_CNN, Y_train_CNN, Y_test_CNN, Y_valid_CNN = get_datasets(saveimgcolor_gffd,imglabels_gffd, valid = True)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]**2))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]**2))
#=======================================================================================================================

#=======================================================================================================================
#Task C
#tests with an SVM
taskidx = 2     #glasses detection task
filetaskidx = 3
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx)

taskidx = 3     # emotion recognition task
filetaskidx = 1
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx)

taskidx = 4     # age identification task
filetaskidx = 2
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx)

taskidx = 5     # human detection task
filetaskidx = 4
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx)

taskidx = 1     # hair color task
filetaskidx = 5
SVM_solve_task(np.reshape(X_train_CNN,(X_train_CNN.shape[0],(X_train_CNN.shape[1]**2)*3)), np.reshape(X_test_CNN,(X_test_CNN.shape[0],(X_test_CNN.shape[1]**2)*3)), Y_train_CNN, Y_test_CNN, taskidx, filetaskidx)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# test with a CNN
# taskidx = 2     #glasses detection task
# filetaskidx = 3
# CNN_solve_task(X_train_CNN, X_test_CNN, X_valid_CNN,Y_train_CNN, Y_test_CNN, Y_valid_CNN,taskidx, filetaskidx)

taskidx = 1
filetaskidx = 5
CNN_solve_task(X_train_CNN, X_test_CNN, X_valid_CNN,Y_train_CNN, Y_test_CNN, Y_valid_CNN,taskidx, filetaskidx)






