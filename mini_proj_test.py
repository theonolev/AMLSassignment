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
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# PATH TO ALL IMAGES
global basedir, image_paths, target_size, img_size, starttime, testimage_paths
basedir = os.path.dirname(os.path.abspath(__file__))
basedir = os.path.join(basedir,'dataset')
print(basedir)
images_dir = os.path.join(basedir,'celeba')
testimages_dir = os.path.join(basedir,'testing_dataset')
labels_filename = os.path.join(basedir,'attribute_list.csv')
img_size = 64

image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
testimage_paths = [os.path.join(testimages_dir, l) for l in os.listdir(testimages_dir)]
target_size = None
starttime = timeit.default_timer()

def face_detect_gffd(saveimg):
    # helper function to detect face if there is one in the picture, based on the HOG technique associated with a pretrained SVM pre implemented in the dlib library
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
    # helper function to detect face if there is one in the picture, based on the structural similarity technique pre implemented in the skimage library
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
    # helper function to detect face if there is one in the picture, based on the haar cascade technique pre implemented in the openCV library
    print("starting haar detection")
    start = timeit.default_timer()
    face_cascade = cv2.CascadeClassifier('C:\\Users\\TheoV\\PycharmProjects\\untitled\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    outliers_pred = np.array([])

    for imgidx in range(len(saveimg)):
        facedetected = face_cascade.detectMultiScale(saveimg[imgidx], 1.1, 5)
        if type(facedetected)is tuple :
            outliers_pred = np.append(outliers_pred, imgidx+1)
    stop = timeit.default_timer()
    print('Time haar: ', stop - start)
    return outliers_pred


def get_real_outliers(imglabels) :
    # find the outliers in the dataset from the labels (if every feature is labelled -1 then example is assumed to be an outlier)
    # the output will be used to compute the accuracy of the face detection methods implemented for task A
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
    # helper function used to compute the accuracy of the different methods for face detection implemented in task A
    truenegarray = [x for x in real if x in pred]
    trueposarray = [x for x in np.arange(1,len(labels)+1) if x not in real and x not in pred]
    falsenegarray = [x for x in pred if x not in real]
    falseposarray = [x for x in real if x not in pred]

    print("true negatives = ", len(truenegarray))
    print("false positives = ", len(falseposarray))
    print("false negatives = ", len(falsenegarray))
    print("true positives = ", len(trueposarray))
    accur = (len(truenegarray)+len(trueposarray))/(len(labels))*100  # accuracy = (true pos + true neg)/overall number of examples
    print("accuracy = ", accur, "%")
    print("\n")


def get_datasets(saveimg, imglabels, trainval = 0.7, testval = 0.3, valid = False):
    # helper function splitting the dataset into training, testing and if required validation set.
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
    # helper function called to create the CNN model, untrained, by stacking up the different layers required
    model = Sequential()
    model.add(Conv2D(5, (3, 3), padding='same', activation='relu', input_shape=(img_size,img_size,3)))  # convolutional layer with activation function included
    model.add(MaxPooling2D(pool_size=(2, 2)))   # pooling layer
    model.add(Dropout(0.25))    #dropout to avoid overfitting

    model.add(Conv2D(10, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())        #flattening the output of the previous layer to feed it to the fully connected layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    if multiclass :     #output of the fully connected layers using softmax to define 1 class only to be labelled 1
        model.add(Dense(7, activation='softmax'))
    else:
        model.add(Dense(2, activation='softmax'))

    return model


def SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx, testimg) :
    #  helper function called to create, train the SVM with the training set, and use test set to compute the accuracy
    print("\nstarting task %i with SVM" % (filetaskidx))
    print("creating SVM")
    clf = svm.SVC(kernel="poly", gamma='scale', degree=3, C=1)      #creates the SVM
    print("training SVM")
    clf.fit(X_train, Y_train[:, taskidx])       #trains the SVM
    print("making predictions")
    predic = clf.predict(X_test)        #get prediction on unknown data
    accur = metrics.balanced_accuracy_score(Y_test[:, taskidx], predic)
    confus = metrics.confusion_matrix(Y_test[:, taskidx], predic)
    print("accuracy = ", accur)
    print("confusion matrix : \n", confus)
    stoptime = timeit.default_timer()
    print("exec time = ", stoptime - starttime)

    predic = clf.predict(testimg)  # get prediction on unknown data

    # create CSV file
    csvtitle = "task_%i.csv" % (filetaskidx)
    with open(os.path.join(basedir, csvtitle), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["average inference accuracy = %f" % (accur)])
        for idx in range(len(predic)):
            row = zip(["%i.png" % (idx+1)], [predic[idx].astype(int)])
            wr.writerows(row)


def CNN_solve_task(X_train_CNN, X_test_CNN, X_valid_CNN,Y_train_CNN, Y_test_CNN, Y_valid_CNN,taskidx, filetaskidx, testimgcolor) :
    #  helper function called to create, compile and train the CNN with the training and validation sets, and plot the accuracy and loss functions
    print("CNN model creation")
    if filetaskidx != 5:            #creates the CNN by stacking up the required layers
        model1 = createModel()
    else :
        model1 = createModel(multiclass = True)
    print("CNN model done")
    batch_size = 200
    epochs = 50
    model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])      #compiles the CNN by assigning it with weights, optimizer and loss functions
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience = 7 , restore_best_weights=True)     #sets a stopping criterion
    callbacks_list = [earlystop]

    if filetaskidx != 5 :
        nb_classes = np.max(Y_train_CNN[:, taskidx].astype(int)) + 1
        Y_train_CNN_task = Y_train_CNN[:,taskidx].clip(min=0)
        Y_valid_CNN_task = Y_valid_CNN[:,taskidx].clip(min=0)
        Y_test_CNN_task = Y_test_CNN[:,taskidx].clip(min=0)
    else :
        nb_classes = np.max(Y_train_CNN[:, taskidx].astype(int)) + 2
        Y_train_CNN_task = Y_train_CNN[:, taskidx] + 1
        Y_valid_CNN_task = Y_valid_CNN[:, taskidx] + 1
        Y_test_CNN_task = Y_test_CNN[:, taskidx] + 1

    # normalize image colour values (from range 0-255 to range 0-1) so it doesn't kill weight in the activation layers
    X_train_CNN = X_train_CNN / 255
    X_test_CNN = X_test_CNN / 255
    X_valid_CNN = X_valid_CNN / 255
    testimgcolor = testimgcolor / 255

    Y_train_CNN_task = np_utils.to_categorical(Y_train_CNN_task, nb_classes)
    Y_valid_CNN_task = np_utils.to_categorical(Y_valid_CNN_task, nb_classes)
    Y_test_CNN_task = np_utils.to_categorical(Y_test_CNN_task, nb_classes)
    history = model1.fit(X_train_CNN, Y_train_CNN_task, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1,
                         validation_data=(X_valid_CNN, Y_valid_CNN_task))       #trains the CNN
    print("\nCNN model evaluation")
    predic_CNN = model1.predict(X_test_CNN)     #get prediction on unknown data
    if filetaskidx != 5:
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN]) - 1
    else :
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN])
    mod_eval = model1.evaluate(X_test_CNN, Y_test_CNN_task)     #evaluate average accuracy of the model based on validation accuracy.
    eval_accur = mod_eval[1]

    confus = metrics.confusion_matrix(Y_test_CNN[:, taskidx], predic_CNN)
    print("accuracy = ", eval_accur)
    print("confusion matrix : \n", confus)
    stoptime = timeit.default_timer()
    print("exec time = ", stoptime - starttime)

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()

    predic_CNN = model1.predict(testimgcolor)  # get prediction on unknown data
    if filetaskidx != 5:
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN]) - 1
    else:
        predic_CNN = np.asarray([np.argmax(y, axis=None, out=None) for y in predic_CNN])

    #create CSV file
    csvtitle = "task_%i.csv" % (filetaskidx)
    with open(os.path.join(basedir, csvtitle), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["average inference accuracy = %f" % (eval_accur)])
        for idx in range(len(predic_CNN)):
            row = zip(["%i.png" % (idx+1)], [predic_CNN[idx].astype(int)])
            wr.writerows(row)

#=======================================================================================================================
#Open all images and save values
if os.path.isfile(os.path.join(basedir,"saveimg.npy")) and os.path.isfile(os.path.join(basedir,"saveimgcolor.npy")) and os.path.isfile(os.path.join(basedir,"testimg.npy")) and os.path.isfile(os.path.join(basedir,"testimgcolor.npy")) :
    saveimg = np.load(os.path.join(basedir,"saveimg.npy"))
    saveimgcolor = np.load(os.path.join(basedir, "saveimgcolor.npy"))
    testimg = np.load(os.path.join(basedir,"testimg.npy"))
    testimgcolor = np.load(os.path.join(basedir, "testimgcolor.npy"))
    #TODO : load testimg and testimgcolor

    print("image array loaded")
elif os.path.isdir(images_dir) :
    saveimg = np.empty([5000, img_size, img_size], dtype=np.uint8)
    saveimgcolor = np.empty([5000, img_size, img_size, 3], dtype=np.uint8)
    testimg = np.empty([100, img_size, img_size], dtype=np.uint8)
    testimgcolor = np.empty([100, img_size, img_size, 3], dtype=np.uint8)
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

    for img_path in testimage_paths:
        file_name = img_path.split('.')[0].split('\\')[-1]
        # load image
        img = image.img_to_array(
            image.load_img(img_path,
                           target_size=target_size,
                           interpolation='bicubic'))
        img = cv2.resize(img, (img_size, img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        testimg[int(file_name)-1] =  gray     #save gray img pixel values to array for later processing
        testimgcolor[int(file_name)-1] =  img
    testimg = testimg.astype(np.uint8)  #convert img to uint8, useful for imshow
    np.save(os.path.join(basedir,"testimg"),testimg)
    testimgcolor = testimgcolor.astype(np.uint8)  # convert img to uint8, useful for imshow
    np.save(os.path.join(basedir, "testimgcolor"), testimgcolor)

    print("image array storing done")
else :
    print("image folder not found")

#open csv file to store labels or load ndarray with labels
if os.path.isfile(os.path.join(basedir,"imglabels.npy")) :
    imglabels = np.load(os.path.join(basedir,"imglabels.npy"))
    print("labels loaded")
else :
    with open(labels_filename,'r') as csvfile:
        labelscsv = csv.reader(csvfile, delimiter=',')
        imglabels = np.zeros([5000, 6])
        counter = 1;
        for row in labelscsv:
            if counter >= 3 : # ignore headers for csv file
                currlabel = np.append(int(row[0]),row[1:])
                imglabels[int(row[0])-1,:] = currlabel
            counter += 1
        imglabels.astype(int)
        np.save(os.path.join(basedir,"imglabels.npy"),imglabels)
        print("labels saved")

if not os.path.isdir(os.path.join(basedir, "outliers")):
    os.mkdir(os.path.join(basedir, "outliers"))
#=======================================================================================================================

#=======================================================================================================================
#Task A: face detection to remove outliers
#tests with different methods for face detection: SSIM, HAAR cascade and HOG => HOG seems most accurate with acc = 92%

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
testimg = np.reshape(testimg,(testimg.shape[0],testimg.shape[1]**2))
#=======================================================================================================================

#=======================================================================================================================
#Task C
#tests with an SVM
taskidx = 2     #glasses detection task
filetaskidx = 3
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx, testimg)

taskidx = 3     # emotion recognition task
filetaskidx = 1
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx, testimg)

taskidx = 4     # age identification task
filetaskidx = 2
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx, testimg)

taskidx = 5     # human detection task
filetaskidx = 4
SVM_solve_task(X_train, X_test, Y_train, Y_test, taskidx, filetaskidx, testimg)
#
# taskidx = 1     # hair color task
# filetaskidx = 5
# SVM_solve_task(np.reshape(X_train_CNN,(X_train_CNN.shape[0],(X_train_CNN.shape[1]**2)*3)), np.reshape(X_test_CNN,(X_test_CNN.shape[0],(X_test_CNN.shape[1]**2)*3)), Y_train_CNN, Y_test_CNN, taskidx, filetaskidx, testimgcolor)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# test with a CNN
taskidx = 1
filetaskidx = 5
CNN_solve_task(X_train_CNN, X_test_CNN, X_valid_CNN,Y_train_CNN, Y_test_CNN, Y_valid_CNN,taskidx, filetaskidx, testimgcolor)






