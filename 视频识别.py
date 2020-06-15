#!/usr/bin/env python3
import os
import re
import sqlite3

import face_recognition

import cv2
from PyQt5 import QtCore

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi

import logging.config

import sys
import numpy as np


from qtpy import QtWidgets, QtGui

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
isexpressionaceRecognition=False
isfaceaceRecognition=False

def load_model():
    """
    加载本地模型
    :return:
    """
    from model import CNN3
    model = CNN3()
    model.load_weights('cnn3_best_weights.h5')
    return model

def generate_faces(face_img, img_size=48):
        """
        将探测到的人脸进行增广
        :param face_img: 灰度化的单个人脸图
        :param img_size: 目标图片大小
        :return:
        """
        import cv2
        import numpy as np
        face_img = face_img / 255.
        face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        resized_images = list()
        resized_images.append(face_img)
        resized_images.append(face_img[2:45, :])
        resized_images.append(face_img[1:47, :])
        resized_images.append(cv2.flip(face_img[:, :], 1))

        for i in range(len(resized_images)):
            resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
            resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
        resized_images = np.array(resized_images)
        return resized_images


en_name=''
def DBfaceid(id):
    global en_name
    conn = sqlite3.connect('FaceBase.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE stu_id=?", (id,))
    result = cursor.fetchall()
    print('查询id',id,'查询结果',result)
    if result:
        en_name = result[0][3]
        print('11', en_name)
    cursor.close()
    conn.close()
    return en_name

def panduan(leng):
    t=0
    for i in range(8):
        if leng[i]==True:
            t+=1
    return t

def load_coding(path='photo'):
    list_dirs = os.walk(path)
    coding=[]
    for root, dirs, files in list_dirs:

        for f in range(len(files)):
            filepath = os.path.join(root, files[f])
            a = face_recognition.load_image_file(filepath)
            a_encodeing = face_recognition.face_encodings(a)[0]
            coding.append(a_encodeing)
        allfiles=files
        return coding,allfiles
coding,allfiles=load_coding()




def show(img):
    global allfiles
    global coding
    print(len(coding))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(32, 32)
    )

    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        g_face = gray[y:y + h, x:x + w]
        faces = generate_faces(g_face)
        expressionresults = model.predict(faces)

        result_sum = np.sum(expressionresults, axis=0).reshape(-1)
        label_index = np.argmax(result_sum, axis=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if isexpressionaceRecognition:
            cv2.putText(img, emotions[label_index], (x, y ), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))

        try:
            unknown_encoding = face_recognition.face_encodings(face)[0]
        except:
            continue
        results = face_recognition.compare_faces(coding,unknown_encoding,tolerance=0.6)
        if panduan(results)>1:
            m=0.01
            while panduan(results)>1:
                results = face_recognition.compare_faces(
                    coding, unknown_encoding, tolerance=0.6-m)
                m+=0.01
        for i in range(len(results)):

            if results[i] == True:
                iswrite=1
                # img = cv2.putText(img, 'name[i]', x + 30, y + 30, (255, 0, 0), 40)
                stu_id=re.findall(r"\d+",allfiles[i])
                print(stu_id)
                en1_name=DBfaceid(stu_id[0])
                cv2.putText(img, en1_name, (x + 30, y + 150), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))

        try:
            cv2.imshow('ssf',img)
            cv2.waitKey(30)
        except:
            pass

    out.write(img)


def recoginition(path):

    cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret == False):
            break
        show(frame)

def init():
    cap = cv2.VideoCapture('./video/555.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video\\test1.avi', fourcc, fps, (width, height))
    return out


emotions = ['anger', 'disgust', 'fear', 'happy', 'happy', 'surprised', 'neutral', 'contempt']
model = load_model()
coding,flies=load_coding()
out=init()



class CoreUI(QMainWindow):


    def __init__(self):
        super(CoreUI, self).__init__()
        loadUi('./ui/video.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(642, 623)

        self.chooseButton.clicked.connect(self.choosefile)

        self.isfaceaceRecognition=False
        self.faceRecognitionCheckBox.clicked[bool].connect(self.faceRecognition)

        self.isexpressionaceRecognition=False
        self.expressionRecognitionCheckBox.clicked[bool].connect(self.expressionRecognition)

        self.startButton.clicked.connect(self.startRecognition)
    def startRecognition(self):
        # if self.filenamelabel.text()=="尚未选择文件":
        #     print('llllllll')
        #     return
        QMessageBox.information(self,"标题","识别需要很久，您确定吗？",QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
        recoginition(self.filenamelabel.text())

    def expressionRecognition(self,p):
        print('表情识别',p)
        global isexpressionaceRecognition
        isexpressionaceRecognition = p
    def faceRecognition(self,p):
        print('人脸识别',p)
        global isfaceaceRecognition
        isfaceaceRecognition = p
    def choosefile(self):
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory="../data/test",
                                                 filter="All Files (*);;Text Files (*.txt)")
        self.filenamelabel.setText(file_name)
        self.filenamelabel.adjustSize();
        print(file_name)



if __name__ == '__main__':
    # logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = CoreUI()
    window.show()
    sys.exit(app.exec())
    pass
