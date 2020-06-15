#!/usr/bin/env python3
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import face_recognition
import numpy as np

import dlib
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox
import webbrowser
import logging.config
import winsound
from configparser import ConfigParser
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QTableWidgetItem, QAbstractItemView
import logging.config
import shutil
import multiprocessing
import cv2
from PyQt5.QtCore import QTimer, QRegExp, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QRegExpValidator, QTextCursor
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMessageBox
from PyQt5.uic import loadUi
import logging
import logging.config
import queue
import threading
import sqlite3
import os
import sys
from datetime import datetime



pre='the face is '
face_name='waiting for identification'
# 自定义数据库记录不存在异常
class RecordNotFound(Exception):
    pass


class DataManageUI(QWidget):
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # 日志信号

    def __init__(self):
        super(DataManageUI, self).__init__()
        loadUi('./ui/DataManage.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(931, 577)

        # 设置tableWidget只读，不允许修改
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # 数据库
        self.database = './FaceBase.db'
        self.datasets = './datasets'
        self.isDbReady = False
        self.initDbButton.clicked.connect(self.initDb)

        # 用户管理
        self.queryUserButton.clicked.connect(self.queryUser)
        self.deleteUserButton.clicked.connect(self.deleteUser)

        # 直方图均衡化
        self.isEqualizeHistEnabled = False
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.enableEqualizeHist(self.equalizeHistCheckBox))

        # 训练人脸数据
        self.trainButton.clicked.connect(self.train)

        # 系统日志
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

        self.Managefirst.toggled.connect(self.firstfunction)
        self.Managefirst.setCheckable(True)


        self.Managesecond.toggled.connect(self.secondfunction)
        self.Managesecond.setCheckable(True)

    def firstfunction(self):
        self.close()
        window = DataRecordUI()
        window.show()

    def secondfunction(self):
        self.close()
        window = CoreUI()
        window.show()

    # 是否执行直方图均衡化
    def enableEqualizeHist(self, equalizeHistCheckBox):
        if equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
        else:
            self.isEqualizeHistEnabled = False

    # 初始化/刷新数据库
    def initDb(self):
        # 刷新前重置tableWidget
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
        try:
            if not os.path.isfile(self.database):
                raise FileNotFoundError

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            res = cursor.execute('SELECT * FROM users')
            for row_index, row_data in enumerate(res):
                self.tableWidget.insertRow(row_index)
                for col_index, col_data in enumerate(row_data):
                    self.tableWidget.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except FileNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.isDbReady = False
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except Exception:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化/刷新数据库失败')
        else:
            cursor.close()
            conn.close()

            self.dbUserCountLcdNum.display(dbUserCount)
            if not self.isDbReady:
                self.isDbReady = True
                self.logQueue.put('Success：数据库初始化完成，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setText('刷新数据库')
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                self.trainButton.setToolTip('')
                self.trainButton.setEnabled(True)
                self.queryUserButton.setToolTip('')
                self.queryUserButton.setEnabled(True)
            else:
                self.logQueue.put('Success：刷新数据库成功，发现用户数：{}'.format(dbUserCount))

    # 查询用户
    def queryUser(self):
        stu_id = self.queryUserLineEdit.text().strip()
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
            ret = cursor.fetchall()
            if not ret:
                raise RecordNotFound
            face_id = ret[0][1]
            cn_name = ret[0][2]
        except RecordNotFound:
            self.queryUserButton.setIcon(QIcon('./icons/error.png'))
            self.queryResultLabel.setText('<font color=red>Error：此用户不存在</font>')
        except Exception as e:
            logging.error('读取数据库异常，无法查询到{}的用户信息'.format(stu_id))
            self.queryResultLabel.clear()
            self.queryUserButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，查询失败')
        else:
            self.queryResultLabel.clear()
            self.queryUserButton.setIcon(QIcon('./icons/success.png'))
            self.stuIDLineEdit.setText(stu_id)
            self.cnNameLineEdit.setText(cn_name)
            self.faceIDLineEdit.setText(str(face_id))
            self.deleteUserButton.setEnabled(True)
        finally:
            cursor.close()
            conn.close()

    # 删除用户
    def deleteUser(self):
        text = '从数据库中删除该用户，同时删除相应人脸数据，<font color=red>该操作不可逆！</font>'
        informativeText = '<b>是否继续？</b>'
        ret = DataManageUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                      QMessageBox.No)

        if ret == QMessageBox.Yes:
            stu_id = self.stuIDLineEdit.text()
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('DELETE FROM users WHERE stu_id=?', (stu_id,))
            except Exception as e:
                cursor.close()
                logging.error('无法从数据库中删除{}'.format(stu_id))
                self.deleteUserButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，删除失败')
            else:
                cursor.close()
                conn.commit()
                if os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):
                    try:
                        shutil.rmtree('{}/stu_{}'.format(self.datasets, stu_id))
                    except Exception as e:
                        logging.error('系统无法删除删除{}/stu_{}'.format(self.datasets, stu_id))
                        self.logQueue.put('Error：删除人脸数据失败，请手动删除{}/stu_{}目录'.format(self.datasets, stu_id))

                text = '你已成功删除学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                informativeText = '<b>请在右侧菜单重新训练人脸数据。</b>'
                DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                self.stuIDLineEdit.clear()
                self.cnNameLineEdit.clear()
                self.faceIDLineEdit.clear()
                self.initDb()
                self.deleteUserButton.setIcon(QIcon('./icons/success.png'))
                self.deleteUserButton.setEnabled(False)
                self.queryUserButton.setIcon(QIcon())
            finally:
                conn.close()

    # 检测人脸
    def detectFace(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.isEqualizeHistEnabled:
            gray = cv2.equalizeHist(gray)
        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(90, 90))

        if (len(faces) == 0):
            return None, None
        (x, y, w, h) = faces[0]
        return gray[y:y + w, x:x + h], faces[0]

    # 准备图片数据
    def prepareTrainingData(self, data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []

        face_id = 1
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        # 遍历人脸库
        for dir_name in dirs:
            if not dir_name.startswith('stu_'):
                continue
            stu_id = dir_name.replace('stu_', '')
            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                ret = cursor.fetchall()
                if not ret:
                    raise RecordNotFound
                cursor.execute('UPDATE users SET face_id=? WHERE stu_id=?', (face_id, stu_id,))
            except RecordNotFound:
                logging.warning('数据库中找不到学号为{}的用户记录'.format(stu_id))
                self.logQueue.put('发现学号为{}的人脸数据，但数据库中找不到相应记录，已忽略'.format(stu_id))
                continue
            subject_dir_path = data_folder_path + '/' + dir_name
            subject_images_names = os.listdir(subject_dir_path)
            for image_name in subject_images_names:
                if image_name.startswith('.'):
                    continue
                image_path = subject_dir_path + '/' + image_name
                image = cv2.imread(image_path)
                face, rect = self.detectFace(image)
                if face is not None:
                    faces.append(face)
                    labels.append(face_id)
            face_id = face_id + 1

        cursor.close()
        conn.commit()
        conn.close()

        return faces, labels

    # 训练人脸数据
    # Reference：https://github.com/informramiz/opencv-face-recognition-python
    def train(self):
        try:
            if not os.path.isdir(self.datasets):
                raise FileNotFoundError

            text = '系统将开始训练人脸数据，界面会暂停响应一段时间，完成后会弹出提示。'
            informativeText = '<b>训练过程请勿进行其它操作，是否继续？</b>'
            ret = DataManageUI.callDialog(QMessageBox.Question, text, informativeText,
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
            if ret == QMessageBox.Yes:
                face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                if not os.path.exists('./recognizer'):
                    os.makedirs('./recognizer')
            faces, labels = self.prepareTrainingData(self.datasets)
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.save('./recognizer/trainingData.yml')
        except FileNotFoundError:
            logging.error('系统找不到人脸数据目录{}'.format(self.datasets))
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('未发现人脸数据目录{}，你可能未进行人脸采集'.format(self.datasets))
        except Exception as e:
            logging.error('遍历人脸库出现异常，训练人脸数据失败')
            self.trainButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：遍历人脸库出现异常，训练失败')
        else:
            text = '<font color=green><b>Success!</b></font> 系统已生成./recognizer/trainingData.yml'
            informativeText = '<b>人脸数据训练完成！</b>'
            DataManageUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
            self.trainButton.setIcon(QIcon('./icons/success.png'))
            self.logQueue.put('Success：人脸数据训练完成')
            self.initDb()

    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    # LOG输出
    def logOutput(self, log):
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    # 系统对话框
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - DataManage')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

class OperationCancel(Exception):
    pass


# 采集过程中出现干扰
class RecordDisturbance(Exception):
    pass

class TrainingDataNotFoundError(FileNotFoundError):
    pass


# 找不到数据库文件
class DatabaseNotFoundError(FileNotFoundError):
    pass


class CoreUI(QMainWindow):
    database = './FaceBase.db'
    trainingData = './recognizer/trainingData.yml'
    cap = cv2.VideoCapture()
    captureQueue = queue.Queue()  # 图像队列
    alarmQueue = queue.LifoQueue()  # 报警队列，后进先出
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG信号

    def __init__(self):
        super(CoreUI, self).__init__()
        loadUi('./ui/Core.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(1161, 623)

        #图像对比相似度

        # self.distinguish('test/unknown.png')


        # 图像捕获
        self.isExternalCameraUsed = False
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.faceProcessingThread = FaceProcessingThread()
        self.startWebcamButton.clicked.connect(self.startWebcam)

        # 数据库
        self.initDbButton.setIcon(QIcon('./icons/warning.png'))
        self.initDbButton.clicked.connect(self.initDb)

        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.updateFrame)

        # 功能开关
        self.faceTrackerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceTracker(self))
        self.faceRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceRecognizer(self))

        self.ExpressionRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableExpressionRecognizer(self))

        # self.panalarmCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enablePanalarm(self))

        # 直方图均衡化
        # self.equalizeHistCheckBox.stateChanged.connect(
        #     lambda: self.faceProcessingThread.enableEqualizeHist(self))
        #
        # # 调试模式
        # self.debugCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enableDebug(self))
        # self.confidenceThresholdSlider.valueChanged.connect(
        #     lambda: self.faceProcessingThread.setConfidenceThreshold(self))
        #
        # # 帮助与支持
        # self.viewGithubRepoButton.clicked.connect(
        #     lambda: webbrowser.open('https://github.com/591751889'))
        # self.contactDeveloperButton.clicked.connect(lambda: webbrowser.open('https://t.me/winterssy'))

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

        # self.faceLabel.setText(face_name)

        # self.Corefirst.toggled.connect(self.firstfunction)
        # self.Corefirst.setCheckable(True)
        #
        # self.Coresecond.toggled.connect(self.secondfunction)
        # self.Coresecond.setCheckable(True)



    def firstfunction(self):
        self.close()
        window = DataRecordUI()
        window.show()

    def secondfunction(self):
        self.close()
        window = DataManageUI()
        window.show()


    # 检查数据库状态
    def initDb(self):
        try:
            if not os.path.isfile(self.database):
                raise DatabaseNotFoundError
            if not os.path.isfile(self.trainingData):
                raise TrainingDataNotFoundError

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DatabaseNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except TrainingDataNotFoundError:
            logging.error('系统找不到已训练的人脸数据{}'.format(self.trainingData))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现已训练的人脸数据文件，请完成训练后继续')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
                self.initDbButton.setIcon(QIcon('./icons/warning.png'))
            else:
                self.logQueue.put('Success：数据库状态正常，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                self.initDbButton.setEnabled(False)
                self.faceRecognizerCheckBox.setToolTip('须先开启人脸跟踪')
                self.faceRecognizerCheckBox.setEnabled(True)
                self.ExpressionRecognizerCheckBox.setEnabled(True)

    # 是否使用外接摄像头
    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    # 打开/关闭摄像头
    def startWebcam(self):


        if not self.cap.isOpened():
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()
            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
            else:
                self.faceProcessingThread.start()  # 启动OpenCV图像处理线程
                self.timer.start(5)  # 启动定时器
                # self.panalarmThread.start()  # 启动报警系统线程
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
                self.startWebcamButton.setText('关闭摄像头')

        else:
            text = '如果关闭摄像头，须重启程序才能再次打开。'
            informativeText = '<b>是否继续？</b>'
            ret = CoreUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

            if ret == QMessageBox.Yes:
                self.faceProcessingThread.stop()
                if self.cap.isOpened():
                    if self.timer.isActive():
                        self.timer.stop()
                    self.cap.release()

                self.realTimeCaptureLabel.clear()
                self.realTimeCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('摄像头已关闭')
                self.startWebcamButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    # 定时器，实时更新画面
    def updateFrame(self):
        if self.cap.isOpened():
            # ret, frame = self.cap.read()
            # if ret:
            #     self.showImg(frame, self.realTimeCaptureLabel)
            if not self.captureQueue.empty():
                captureData = self.captureQueue.get()
                realTimeFrame = captureData.get('realTimeFrame')
                self.displayImage(realTimeFrame, self.realTimeCaptureLabel)

    # 显示图片
    def displayImage(self, img, qlabel):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))
        qlabel.setScaledContents(True)  # 图片自适应大小





    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    # LOG输出
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    # 系统对话框
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - Core')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    # 窗口关闭事件，关闭OpenCV线程、定时器、摄像头
    def closeEvent(self, event):
        if self.faceProcessingThread.isRunning:
            self.faceProcessingThread.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()




# OpenCV线程
class FaceProcessingThread(QThread):
    def __init__(self):
        super(FaceProcessingThread, self).__init__()
        self.isRunning = True

        self.isFaceTrackerEnabled = True
        self.isFaceRecognizerEnabled = False

        self.isExpressionRecognizerEnabled=False


        self.isDebugMode = False
        self.confidenceThreshold = 50
        self.autoAlarmThreshold = 65

        self.isEqualizeHistEnabled = False

        self.coding = []
        self.load_coding()
        self.model=self.load_model()
        self.model.summary()
        print('初始化完成')


    def load_model(self):
        """
        加载本地模型
        :return:
        """
        from model import CNN3
        model = CNN3()
        model.load_weights('cnn3_best_weights.h5')
        return model

    def generate_faces(self,face_img, img_size=48):
        """
        将探测到的人脸进行增广
        :param face_img: 灰度化的单个人脸图
        :param img_size: 目标图片大小
        :return:
        """

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
    def load_coding(self, path='photo'):
        list_dirs = os.walk(path)
        for root, dirs, files in list_dirs:

            for f in range(len(files)):
                filepath = os.path.join(root, files[f])
                a = face_recognition.load_image_file(filepath)
                a_encodeing = face_recognition.face_encodings(a)[0]
                self.coding.append(a_encodeing)
            self.files=files
            print(files)

    def DBstu_id(self,id=1607084233):
        conn = sqlite3.connect('FaceBase.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE stu_id=?", (id,))
        result = cursor.fetchall()

        cursor.close()
        conn.close()
        return result

    def distinguish(self, filepath='test/unknown.jpg'):
        try:
            image = face_recognition.load_image_file(filepath)
            img_encoding = face_recognition.face_encodings(image)[0]
            results = face_recognition.compare_faces(self.coding, img_encoding, tolerance=0.5)
            print('result',results)
            ischange=False
            global face_name
            for i in range(len(results)):
                if results[i]==True:
                    print('i=',i)
                    ischange=True
                    string = self.files[i]
                    t = re.findall(r"\d+", string)
                    print(t,self.DBstu_id(t[0]))
                    info=self.DBstu_id(t[0])
                    print(info[0])
                    global face_name
                    face_name=info[0][3]
            if ischange==False:
                face_name='unknown'

        except:
            pass


    # 是否开启人脸跟踪
    def enableFaceTracker(self, coreUI):
        if coreUI.faceTrackerCheckBox.isChecked():
            self.isFaceTrackerEnabled = True
            coreUI.statusBar().showMessage('人脸跟踪：开启')
        else:
            self.isFaceTrackerEnabled = False
            coreUI.statusBar().showMessage('人脸跟踪：关闭')

    # 是否开启人脸识别
    def enableFaceRecognizer(self, coreUI):
        if coreUI.faceRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isFaceRecognizerEnabled = True
                coreUI.statusBar().showMessage('人脸识别：开启')
            else:
                CoreUI.logQueue.put('Error：操作失败，请先开启人脸跟踪')
                coreUI.faceRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.faceRecognizerCheckBox.setChecked(False)
        else:
            self.isFaceRecognizerEnabled = False
            coreUI.statusBar().showMessage('人脸识别：关闭')

    def enableExpressionRecognizer(self,coreUI):
        print('调用表情识别')
        if coreUI.ExpressionRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isExpressionRecognizerEnabled = True
                coreUI.statusBar().showMessage('表情识别：开启')
            else:
                CoreUI.logQueue.put('Error：操作失败，请先开启人脸跟踪')
                coreUI.ExpressionRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.ExpressionRecognizerCheckBox.setChecked(False)
        else:
            self.isExpressionRecognizerEnabled = False
            coreUI.statusBar().showMessage('表情识别：关闭')






    def run(self):
        faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        # 帧数、人脸ID初始化
        frameCounter = 0
        currentFaceID = 0

        frameCount=0

        # 人脸跟踪器字典初始化
        faceTrackers = {}

        isTrainingDataLoaded = False
        isDbConnected = False

        while self.isRunning:
            if CoreUI.cap.isOpened():
                ret, frame = CoreUI.cap.read()
                # frame=cv2.imread('123.jpg')
                frameCount=frameCount+1
                # print(frameCount)
                global face_name

                    # face_name
                cv2.putText(frame, 'this is '+face_name, (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 是否执行直方图均衡化
                # if self.isEqualizeHistEnabled:
                #     gray = cv2.equalizeHist(gray)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

                # 预加载数据文件
                if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(CoreUI.trainingData)
                    isTrainingDataLoaded = True
                if not isDbConnected and os.path.isfile(CoreUI.database):
                    conn = sqlite3.connect(CoreUI.database)
                    cursor = conn.cursor()
                    isDbConnected = True

                captureData = {}
                realTimeFrame = frame.copy()
                alarmSignal = {}

                # 人脸跟踪
                # Reference：https://github.com/gdiepen/face-recognition
                if self.isFaceTrackerEnabled:

                    # 要删除的人脸跟踪器列表初始化
                    fidsToDelete = []

                    for fid in faceTrackers.keys():
                        # 实时跟踪
                        trackingQuality = faceTrackers[fid].update(realTimeFrame)
                        # 如果跟踪质量过低，删除该人脸跟踪器
                        if trackingQuality < 7:
                            fidsToDelete.append(fid)

                    # 删除跟踪质量过低的人脸跟踪器
                    for fid in fidsToDelete:
                        faceTrackers.pop(fid, None)

                    for (_x, _y, _w, _h) in faces:
                        isKnown = False

                        if self.isFaceRecognizerEnabled:
                            cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (232, 138, 30), 2)
                            face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])

                            logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))

                            if self.isDebugMode:
                                CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                            # 从数据库中获取识别人脸的身份信息
                            try:
                                cursor.execute("SELECT * FROM users WHERE face_id=?", (face_id,))
                                result = cursor.fetchall()
                                if result:
                                    en_name = result[0][3]

                                    print(en_name,confidence)
                                else:
                                    raise Exception
                            except Exception as e:
                                logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                en_name = ''

                            # 若置信度评分小于置信度阈值，认为是可靠识别
                            if confidence < self.confidenceThreshold:
                                isKnown = True
                                # cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #             (0, 97, 255), 2)
                                face_name=en_name
                            else:

                                if frameCount % 50 == 0 or frameCount>50:
                                    print('对比重复度')
                                    frameCount = 0
                                    cv2.imwrite('test/unknown.jpg',frame)

                                    self.distinguish('test/unknown.jpg')

                        if self.isExpressionRecognizerEnabled:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
                            for (_x, _y, _w, _h) in faces:
                                face = self.generate_faces(gray[_y:_y + _h, _x:_x + _w])
                                results = self.model.predict(face)
                            print(results)
                            print('shishi'+results)
                            pass
                        # 帧数自增
                        frameCounter += 1

                        # 每读取10帧，检测跟踪器的人脸是否还在当前画面内
                        if frameCounter % 10 == 0:
                            # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                            # 而dlib人脸跟踪器要求的是int类型
                            x = int(_x)
                            y = int(_y)
                            w = int(_w)
                            h = int(_h)

                            # 计算中心点
                            x_bar = x + 0.5 * w
                            y_bar = y + 0.5 * h

                            # matchedFid表征当前检测到的人脸是否已被跟踪
                            matchedFid = None

                            for fid in faceTrackers.keys():
                                # 获取人脸跟踪器的位置
                                # tracked_position 是 dlib.drectangle 类型，用来表征图像的矩形区域，坐标是浮点数
                                tracked_position = faceTrackers[fid].get_position()
                                # 浮点数取整
                                t_x = int(tracked_position.left())
                                t_y = int(tracked_position.top())
                                t_w = int(tracked_position.width())
                                t_h = int(tracked_position.height())

                                # 计算人脸跟踪器的中心点
                                t_x_bar = t_x + 0.5 * t_w
                                t_y_bar = t_y + 0.5 * t_h

                                # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                                # 说明当前人脸已被跟踪
                                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                        (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                    matchedFid = fid

                            # 如果当前检测到的人脸是陌生人脸且未被跟踪
                            if not isKnown and matchedFid is None:
                                # 创建一个人脸跟踪器
                                tracker = dlib.correlation_tracker()
                                # 锁定跟踪范围
                                tracker.start_track(realTimeFrame, dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                                # 将该人脸跟踪器分配给当前检测到的人脸
                                faceTrackers[currentFaceID] = tracker
                                # 人脸ID自增
                                currentFaceID += 1

                    # 使用当前的人脸跟踪器，更新画面，输出跟踪结果
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # 在跟踪帧中圈出人脸
                        # cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                        # cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        #             2)

                captureData['originFrame'] = frame
                captureData['realTimeFrame'] = realTimeFrame
                CoreUI.captureQueue.put(captureData)

            else:
                continue

    # 停止OpenCV线程
    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()

class DataRecordUI(QWidget):
    receiveLogSignal = pyqtSignal(str)

    def __init__(self):
        super(DataRecordUI, self).__init__()
        loadUi('./ui/DataRecord.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(1011, 601)

        # OpenCV
        self.cap = cv2.VideoCapture()
        self.faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        self.logQueue = queue.Queue()  # 日志队列

        # 图像捕获
        self.isExternalCameraUsed = False
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.startWebcamButton.toggled.connect(self.startWebcam)
        self.startWebcamButton.setCheckable(True)

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        # 人脸检测
        self.isFaceDetectEnabled = False
        self.enableFaceDetectButton.toggled.connect(self.enableFaceDetect)
        self.enableFaceDetectButton.setCheckable(True)

        # 数据库
        self.database = './FaceBase.db'
        self.datasets = './datasets'
        self.isDbReady = False
        self.initDbButton.setIcon(QIcon('./icons/warning.png'))
        self.initDbButton.clicked.connect(self.initDb)

        # 用户信息
        self.isUserInfoReady = False
        self.userInfo = {'stu_id': '', 'cn_name': '', 'en_name': ''}
        self.addOrUpdateUserInfoButton.clicked.connect(self.addOrUpdateUserInfo)
        self.migrateToDbButton.clicked.connect(self.migrateToDb)

        # 人脸采集
        self.startFaceRecordButton.clicked.connect(lambda: self.startFaceRecord(self.startFaceRecordButton))
        # self.startFaceRecordButton.setCheckable(True)
        self.faceRecordCount = 0
        self.minFaceRecordCount = 19
        self.isFaceDataReady = False
        self.isFaceRecordEnabled = False
        self.enableFaceRecordButton.clicked.connect(self.enableFaceRecord)

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()


        #修改
        self.Recordfirst.toggled.connect(self.firstfunction)
        self.Recordfirst.setCheckable(True)

        self.Recordsecond.toggled.connect(self.secondfunction)
        self.Recordsecond.setCheckable(True)


    def secondfunction(self):
        self.close()
        window = CoreUI()
        window.show()

        logging.config.fileConfig('./config/logging.cfg')

        # app = QApplication(sys.argv)


    def firstfunction(self):
        self.close()
        logging.config.fileConfig('./config/logging.cfg')
        # app = QApplication(sys.argv)
        window = DataManageUI()
        window.show()
        # sys.exit(app.exec())

    # 是否使用外接摄像头
    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    # 打开/关闭摄像头
    def startWebcam(self, status):
        if status:
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()

            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
                self.startWebcamButton.setChecked(False)
            else:
                self.startWebcamButton.setText('关闭摄像头')
                self.enableFaceDetectButton.setEnabled(True)
                self.timer.start(5)
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
        else:
            if self.cap.isOpened():
                if self.timer.isActive():
                    self.timer.stop()
                self.cap.release()
                self.faceDetectCaptureLabel.clear()
                self.faceDetectCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('打开摄像头')
                self.enableFaceDetectButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    # 开启/关闭人脸检测
    def enableFaceDetect(self, status):
        if self.cap.isOpened():
            if status:
                self.enableFaceDetectButton.setText('关闭人脸检测')
                self.isFaceDetectEnabled = True
            else:
                self.enableFaceDetectButton.setText('开启人脸检测')
                self.isFaceDetectEnabled = False

    # 采集当前捕获帧
    def enableFaceRecord(self):
        if not self.isFaceRecordEnabled:
            self.isFaceRecordEnabled = True

    # 开始/结束采集人脸数据
    def startFaceRecord(self, startFaceRecordButton):
        if startFaceRecordButton.text() == '开始采集人脸数据':
            if self.isFaceDetectEnabled:
                if self.isUserInfoReady:
                    self.addOrUpdateUserInfoButton.setEnabled(False)
                    if not self.enableFaceRecordButton.isEnabled():
                        self.enableFaceRecordButton.setEnabled(True)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.startFaceRecordButton.setText('结束当前人脸采集')
                else:
                    self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.startFaceRecordButton.setChecked(False)
                    self.logQueue.put('Error：操作失败，系统未检测到有效的用户信息')
            else:
                self.startFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：操作失败，请开启人脸检测')
        else:
            if self.faceRecordCount < self.minFaceRecordCount:
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，采集数据过少会导致较大的识别误差。'.format(self.faceRecordCount)
                informativeText = '<b>请至少采集 <font color=red>{}</font> 帧图像。</b>'.format(self.minFaceRecordCount)
                DataRecordUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

            else:
                text = '系统当前采集了 <font color=blue>{}</font> 帧图像，继续采集可以提高识别准确率。'.format(self.faceRecordCount)
                informativeText = '<b>你确定结束当前人脸采集吗？</b>'
                ret = DataRecordUI.callDialog(QMessageBox.Question, text, informativeText,
                                              QMessageBox.Yes | QMessageBox.No,
                                              QMessageBox.No)

                if ret == QMessageBox.Yes:
                    self.isFaceDataReady = True
                    if self.isFaceRecordEnabled:
                        self.isFaceRecordEnabled = False
                    self.enableFaceRecordButton.setEnabled(False)
                    self.enableFaceRecordButton.setIcon(QIcon())
                    self.startFaceRecordButton.setText('开始采集人脸数据')
                    self.startFaceRecordButton.setEnabled(False)
                    self.startFaceRecordButton.setIcon(QIcon())
                    self.migrateToDbButton.setEnabled(True)

    # 定时器，实时更新画面
    def updateFrame(self):
        ret, frame = self.cap.read()
        # self.image = cv2.flip(self.image, 1)
        if ret:
            self.displayImage(frame)

            if self.isFaceDetectEnabled:
                detected_frame = self.detectFace(frame)
                self.displayImage(detected_frame)
            else:
                self.displayImage(frame)

    # 检测人脸
    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

        stu_id = self.userInfo.get('stu_id')

        for (x, y, w, h) in faces:
            if self.isFaceRecordEnabled:
                try:
                    if not os.path.exists('{}/stu_{}'.format(self.datasets, stu_id)):
                        os.makedirs('{}/stu_{}'.format(self.datasets, stu_id))
                    if len(faces) > 1:
                        raise RecordDisturbance

                    cv2.imwrite('{}/stu_{}/img.{}.jpg'.format(self.datasets, stu_id, self.faceRecordCount + 1),
                                gray[y - 20:y + h + 20, x - 20:x + w + 20])
                    cv2.imwrite('photo/{}.jpg'.format(stu_id), frame)
                    # os.rename('photo/before.jpg','photo'+self.stu_id)
                except RecordDisturbance:
                    self.isFaceRecordEnabled = False
                    logging.error('检测到多张人脸或环境干扰')
                    self.logQueue.put('Warning：检测到多张人脸或环境干扰，请解决问题后继续')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/warning.png'))
                    continue
                except Exception as e:
                    logging.error('写入人脸图像文件到计算机过程中发生异常')
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/error.png'))
                    self.logQueue.put('Error：无法保存人脸图像，采集当前捕获帧失败')
                else:
                    self.enableFaceRecordButton.setIcon(QIcon('./icons/success.png'))
                    self.faceRecordCount = self.faceRecordCount + 1
                    self.isFaceRecordEnabled = False
                    self.faceRecordCountLcdNum.display(self.faceRecordCount)
            cv2.rectangle(frame, (x - 5, y - 10), (x + w + 5, y + h + 10), (0, 0, 255), 2)

        return frame

    # 显示图像
    def displayImage(self, img):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.faceDetectCaptureLabel.setPixmap(QPixmap.fromImage(outImage))
        self.faceDetectCaptureLabel.setScaledContents(True)

    # 初始化数据库
    def initDb(self):
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()
        try:
            # 检测人脸数据目录是否存在，不存在则创建
            if not os.path.isdir(self.datasets):
                os.makedirs(self.datasets)

            # 查询数据表是否存在，不存在则创建
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              stu_id VARCHAR(10) PRIMARY KEY NOT NULL,
                              face_id INTEGER DEFAULT -1,
                              cn_name VARCHAR(10) NOT NULL,
                              en_name VARCHAR(16) NOT NULL,
                              created_time DATE DEFAULT (date('now','localtime'))
                              )
                          ''')
            # 查询数据表记录数
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.isDbReady = False
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：初始化数据库失败')
        else:
            self.isDbReady = True
            self.dbUserCountLcdNum.display(dbUserCount)
            self.logQueue.put('Success：数据库初始化完成')
            self.initDbButton.setIcon(QIcon('./icons/success.png'))
            self.initDbButton.setEnabled(False)
            self.addOrUpdateUserInfoButton.setEnabled(True)
        finally:
            cursor.close()
            conn.commit()
            conn.close()

    # 增加/修改用户信息
    def addOrUpdateUserInfo(self):
        self.userInfoDialog = UserInfoDialog()

        stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
            'en_name')
        self.stu_id=self.userInfo.get('stu_id')
        self.userInfoDialog.stuIDLineEdit.setText(stu_id)
        self.userInfoDialog.cnNameLineEdit.setText(cn_name)
        self.userInfoDialog.enNameLineEdit.setText(en_name)

        self.userInfoDialog.okButton.clicked.connect(self.checkToApplyUserInfo)
        self.userInfoDialog.exec()

    # 校验用户信息并提交
    def checkToApplyUserInfo(self):
        if not (self.userInfoDialog.stuIDLineEdit.hasAcceptableInput() and
                self.userInfoDialog.cnNameLineEdit.hasAcceptableInput() and
                self.userInfoDialog.enNameLineEdit.hasAcceptableInput()):
            self.userInfoDialog.msgLabel.setText('<font color=red>你的输入有误，提交失败，请检查并重试！</font>')
        else:
            # 获取用户输入
            self.userInfo['stu_id'] = self.userInfoDialog.stuIDLineEdit.text().strip()
            self.userInfo['cn_name'] = self.userInfoDialog.cnNameLineEdit.text().strip()
            self.userInfo['en_name'] = self.userInfoDialog.enNameLineEdit.text().strip()

            # 信息确认
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            self.stuIDLineEdit.setText(stu_id)
            self.cnNameLineEdit.setText(cn_name)
            self.enNameLineEdit.setText(en_name)

            self.isUserInfoReady = True
            if not self.startFaceRecordButton.isEnabled():
                self.startFaceRecordButton.setEnabled(True)
            self.migrateToDbButton.setIcon(QIcon())

            # 关闭对话框
            self.userInfoDialog.close()

    # 同步用户信息到数据库
    def migrateToDb(self):
        if self.isFaceDataReady:
            stu_id, cn_name, en_name = self.userInfo.get('stu_id'), self.userInfo.get('cn_name'), self.userInfo.get(
                'en_name')
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            try:
                cursor.execute('SELECT * FROM users WHERE stu_id=?', (stu_id,))
                if cursor.fetchall():
                    text = '数据库已存在学号为 <font color=blue>{}</font> 的用户记录。'.format(stu_id)
                    informativeText = '<b>是否覆盖？</b>'
                    ret = DataRecordUI.callDialog(QMessageBox.Warning, text, informativeText,
                                                  QMessageBox.Yes | QMessageBox.No)

                    if ret == QMessageBox.Yes:
                        # 更新已有记录
                        cursor.execute('UPDATE users SET cn_name=?, en_name=? WHERE stu_id=?',
                                       (cn_name, en_name, stu_id,))
                    else:
                        raise OperationCancel  # 记录取消覆盖操作
                else:
                    # 插入新记录
                    cursor.execute('INSERT INTO users (stu_id, cn_name, en_name) VALUES (?, ?, ?)',
                                   (stu_id, cn_name, en_name,))

                cursor.execute('SELECT Count(*) FROM users')
                result = cursor.fetchone()
                dbUserCount = result[0]
            except OperationCancel:
                pass
            except Exception as e:
                logging.error('读写数据库异常，无法向数据库插入/更新记录')
                self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))
                self.logQueue.put('Error：读写数据库异常，同步失败')
            else:
                text = '<font color=blue>{}</font> 已添加/更新到数据库。'.format(stu_id)
                informativeText = '<b><font color=blue>{}</font> 的人脸数据采集已完成！</b>'.format(cn_name)
                DataRecordUI.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)

                # 清空用户信息缓存
                for key in self.userInfo.keys():
                    self.userInfo[key] = ''
                self.isUserInfoReady = False

                self.faceRecordCount = 0
                self.isFaceDataReady = False
                self.faceRecordCountLcdNum.display(self.faceRecordCount)
                self.dbUserCountLcdNum.display(dbUserCount)

                # 清空历史输入
                self.stuIDLineEdit.clear()
                self.cnNameLineEdit.clear()
                self.enNameLineEdit.clear()
                self.migrateToDbButton.setIcon(QIcon('./icons/success.png'))

                # 允许继续增加新用户
                self.addOrUpdateUserInfoButton.setEnabled(True)
                self.migrateToDbButton.setEnabled(False)

            finally:
                cursor.close()
                conn.commit()
                conn.close()
        else:
            self.logQueue.put('Error：操作失败，你尚未完成人脸数据采集')
            self.migrateToDbButton.setIcon(QIcon('./icons/error.png'))

    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    # LOG输出
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    # 系统对话框
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - DataRecord')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    # 窗口关闭事件，关闭定时器、摄像头
    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


# 用户信息填写对话框
class UserInfoDialog(QDialog):
    def __init__(self):
        super(UserInfoDialog, self).__init__()
        loadUi('./ui/UserInfoDialog.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(425, 300)

        # 使用正则表达式限制用户输入
        stu_id_regx = QRegExp('^[0-9]{10}$')
        stu_id_validator = QRegExpValidator(stu_id_regx, self.stuIDLineEdit)
        self.stuIDLineEdit.setValidator(stu_id_validator)

        cn_name_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        cn_name_validator = QRegExpValidator(cn_name_regx, self.cnNameLineEdit)
        self.cnNameLineEdit.setValidator(cn_name_validator)

        en_name_regx = QRegExp('^[ A-Za-z]{1,16}$')
        en_name_validator = QRegExpValidator(en_name_regx, self.enNameLineEdit)
        self.enNameLineEdit.setValidator(en_name_validator)

if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    window = DataRecordUI()
    window.show()
    sys.exit(app.exec())
