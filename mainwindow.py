# \file    mainwindow.py
# \author  IDS Imaging Development Systems GmbH
# \date    2021-01-15
# \since   1.2.0
#
# \version 1.1.1
#
# Copyright (C) 2021, IDS Imaging Development Systems GmbH.
#
# The information in this document is subject to change without notice
# and should not be construed as a commitment by IDS Imaging Development Systems GmbH.
# IDS Imaging Development Systems GmbH does not assume any responsibility for any errors
# that may appear in this document.
#
# This document, or source code, is provided solely as an example of how to utilize
# IDS Imaging Development Systems GmbH software libraries in a sample application.
# IDS Imaging Development Systems GmbH does not assume any responsibility
# for the use or reliability of any portion of this document.
#
# General permission to copy or modify is hereby granted.

import sys

from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QMainWindow, QMessageBox, QWidget
from PySide2.QtGui import QImage
from PySide2.QtCore import Qt, Slot, QTimer

from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import cv2
import matplotlib.pyplot as plt
import numpy as np
from display import Display
from scipy import ndimage

VERSION = "1.0.1"
FPS_LIMIT = 30


class MainWindow(QMainWindow):
    yy=1500
    h =200
    xx=1500
    w=200

    markx = 250
    marky = 250

    markx1 = 250
    marky1 = 250

    def draw_circle(self,event,x,y,flags,param):
        # global mouseX,mouseY,xx,yy,markx1,marky1
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(img,(x,y),100,(255,0,0),-1)
            self.mouseX,self.mouseY = x,y
            self.xx = x*5
            self.yy = y*5
            self.markx1 = x
            self.marky1 = y
            print("mouse position"+str(x)+"   "+str(y))

    def draw_marker(self,event,x,y,flags,param):
        # global mouseX,mouseY,markx,marky
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(img,(x,y),100,(255,0,0),-1)
            self.mouseX,self.mouseY = x,y
            self.markx = x
            self.marky = y
            print("mouse position"+str(x)+"   "+str(y))

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self.widget = QWidget(self)
        self.__layout = QVBoxLayout()
        self.widget.setLayout(self.__layout)
        self.setCentralWidget(self.widget)

        self.__device = None
        self.__nodemap_remote_device = None
        self.__datastream = None

        self.__display = None
        self.__acquisition_timer = QTimer()
        self.__frame_counter = 0
        self.__error_counter = 0
        self.__acquisition_running = False

        self.__label_infos = None
        self.__label_version = None
        self.__label_aboutqt = None

        # initialize peak library
        ids_peak.Library.Initialize()

        if self.__open_device():
            try:
                # Create a display for the camera image
                self.__display = Display()
                self.__layout.addWidget(self.__display)
                if not self.__start_acquisition():
                    QMessageBox.critical(self, "Unable to start acquisition!", QMessageBox.Ok)
            except Exception as e:
                QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)

        else:
            self.__destroy_all()
            sys.exit(0)

        self.__create_statusbar()

        self.setMinimumSize(700, 500)
        
    def __del__(self):
        self.__destroy_all()

    def __destroy_all(self):
        # Stop acquisition
        self.__stop_acquisition()

        # Close device and peak library
        self.__close_device()
        ids_peak.Library.Close()

    def __open_device(self):
        try:
            # Create instance of the device manager
            device_manager = ids_peak.DeviceManager.Instance()

            # Update the device manager
            device_manager.Update()

            # Return if no device was found
            if device_manager.Devices().empty():
                QMessageBox.critical(self, "Error", "No device found!", QMessageBox.Ok)
                return False

            # Open the first openable device in the managers device list
            for device in device_manager.Devices():
                if device.IsOpenable():
                    self.__device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                    break

            # Return if no device could be opened
            if self.__device is None:
                QMessageBox.critical(self, "Error", "Device could not be opened!", QMessageBox.Ok)
                return False

            # Open standard data stream
            datastreams = self.__device.DataStreams()
            if datastreams.empty():
                QMessageBox.critical(self, "Error", "Device has no DataStream!", QMessageBox.Ok)
                self.__device = None
                return False

            self.__datastream = datastreams[0].OpenDataStream()

            # Get nodemap of the remote device for all accesses to the genicam nodemap tree
            self.__nodemap_remote_device = self.__device.RemoteDevice().NodeMaps()[0]

            # To prepare for untriggered continuous image acquisition, load the default user set if available and
            # wait until execution is finished
            try:
                self.__nodemap_remote_device.FindNode("UserSetSelector").SetCurrentEntry("Default")
                self.__nodemap_remote_device.FindNode("UserSetLoad").Execute()
                self.__nodemap_remote_device.FindNode("UserSetLoad").WaitUntilDone()
            except ids_peak.Exception:
                # Userset is not available
                pass

            # Get the payload size for correct buffer allocation
            payload_size = self.__nodemap_remote_device.FindNode("PayloadSize").Value()

            # Get minimum number of buffers that must be announced
            buffer_count_max = self.__datastream.NumBuffersAnnouncedMinRequired()

            # Allocate and announce image buffers and queue them
            for i in range(buffer_count_max):
                buffer = self.__datastream.AllocAndAnnounceBuffer(payload_size)
                self.__datastream.QueueBuffer(buffer)

            return True
        except ids_peak.Exception as e:
            QMessageBox.critical(self, "Exception", str(e), QMessageBox.Ok)

        return False

    def __close_device(self):
        """
        Stop acquisition if still running and close datastream and nodemap of the device
        """
        # Stop Acquisition in case it is still running
        self.__stop_acquisition()

        # If a datastream has been opened, try to revoke its image buffers
        if self.__datastream is not None:
            try:
                for buffer in self.__datastream.AnnouncedBuffers():
                    self.__datastream.RevokeBuffer(buffer)
            except Exception as e:
                QMessageBox.information(self, "Exception", str(e), QMessageBox.Ok)

    def __start_acquisition(self):
        """
        Start Acquisition on camera and start the acquisition timer to receive and display images

        :return: True/False if acquisition start was successful
        """
        # Check that a device is opened and that the acquisition is NOT running. If not, return.
        if self.__device is None:
            return False
        if self.__acquisition_running is True:
            return True

        # Get the maximum framerate possible, limit it to the configured FPS_LIMIT. If the limit can't be reached, set
        # acquisition interval to the maximum possible framerate
        try:
            max_fps = self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").Maximum()
            target_fps = min(max_fps, FPS_LIMIT)
            self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").SetValue(target_fps)
        except ids_peak.Exception:
            # AcquisitionFrameRate is not available. Unable to limit fps. Print warning and continue on.
            QMessageBox.warning(self, "Warning",
                                "Unable to limit fps, since the AcquisitionFrameRate Node is"
                                " not supported by the connected camera. Program will continue without limit.")

        # Setup acquisition timer accordingly
        self.__acquisition_timer.setInterval((1 / target_fps) * 1000)
        self.__acquisition_timer.setSingleShot(False)
        self.__acquisition_timer.timeout.connect(self.on_acquisition_timer)

        try:
            # Lock critical features to prevent them from changing during acquisition
            self.__nodemap_remote_device.FindNode("TLParamsLocked").SetValue(1)

            # Start acquisition on camera
            self.__datastream.StartAcquisition()
            self.__nodemap_remote_device.FindNode("AcquisitionStart").Execute()
            self.__nodemap_remote_device.FindNode("AcquisitionStart").WaitUntilDone()
        except Exception as e:
            print("Exception: " + str(e))
            return False

        # Start acquisition timer
        self.__acquisition_timer.start()
        self.__acquisition_running = True

        return True

    def __stop_acquisition(self):
        """
        Stop acquisition timer and stop acquisition on camera
        :return:
        """
        # Check that a device is opened and that the acquisition is running. If not, return.
        if self.__device is None or self.__acquisition_running is False:
            return

        # Otherwise try to stop acquisition
        try:
            # Stop acquisition timer and camera acquisition
            self.__acquisition_timer.stop()
            remote_nodemap = self.__device.RemoteDevice().NodeMaps()[0]
            remote_nodemap.FindNode("AcquisitionStop").Execute()

            # Stop and flush datastream
            self.__datastream.KillWait()
            self.__datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self.__datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            self.__acquisition_running = False

            # Unlock parameters after acquisition stop
            if self.__nodemap_remote_device is not None:
                try:
                    self.__nodemap_remote_device.FindNode("TLParamsLocked").SetValue(0)
                except Exception as e:
                    QMessageBox.information(self, "Exception", str(e), QMessageBox.Ok)

        except Exception as e:
            QMessageBox.information(self, "Exception", str(e), QMessageBox.Ok)

    def __create_statusbar(self):
        status_bar = QWidget(self.centralWidget())
        status_bar_layout = QHBoxLayout()
        status_bar_layout.setContentsMargins(0, 0, 0, 0)

        self.__label_infos = QLabel(status_bar)
        self.__label_infos.setAlignment(Qt.AlignLeft)
        status_bar_layout.addWidget(self.__label_infos)
        status_bar_layout.addStretch()

        self.__label_version = QLabel(status_bar)
        self.__label_version.setText("simple_live_qtwidgets v" + VERSION)
        self.__label_version.setAlignment(Qt.AlignRight)
        status_bar_layout.addWidget(self.__label_version)

        self.__label_aboutqt = QLabel(status_bar)
        self.__label_aboutqt.setObjectName("aboutQt")
        self.__label_aboutqt.setText("<a href='#aboutQt'>About Qt</a>")
        self.__label_aboutqt.setAlignment(Qt.AlignRight)
        self.__label_aboutqt.linkActivated.connect(self.on_aboutqt_link_activated)
        status_bar_layout.addWidget(self.__label_aboutqt)
        status_bar.setLayout(status_bar_layout)

        self.__layout.addWidget(status_bar)

    def update_counters(self):
        """
        This function gets called when the frame and error counters have changed
        :return:
        """
        self.__label_infos.setText("Acquired: " + str(self.__frame_counter) + ", Errors: " + str(self.__error_counter))

    @Slot()
    def on_acquisition_timer(self):
        """
        This function gets called on every timeout of the acquisition timer
        """
        try:
            # Get buffer from device's datastream
            buffer = self.__datastream.WaitForFinishedBuffer(5000)

            # Create IDS peak IPL image for debayering and convert it to RGBa8 format
            ipl_image = ids_peak_ipl.Image_CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height()
            )
            converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_BGRa8)
             # rgb_image = cv2.cvtColor(converted_ipl_image, cv2.COLOR_BGR2RGB)
            # Queue buffer so that it can be used again
            self.__datastream.QueueBuffer(buffer)

            # Get raw image data from converted image and construct a QImage from it
            image_np_array = converted_ipl_image.get_numpy_3D()
            # cv2.imshow("crop", image_np_array)
            # cv2.setMouseCallback('crop',self.draw_marker)
            self.impro(image_np_array,converted_ipl_image.Width(),converted_ipl_image.Height())
            # print(image_np_array)
            # cv2.cvtColor(image_np_array, cv2.COLOR_GRAY2RGB, image_np_array)    
           
            image = QImage(image_np_array,
                           converted_ipl_image.Width(), converted_ipl_image.Height(),
                           QImage.Format_RGB32)
            # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image_np_array)
            # cv2.imshow("crop", img)
            # Make an extra copy of the QImage to make sure that memory is copied and can't get overwritten later on
            image_cpy = image.copy()

            # Emit signal that the image is ready to be displayed
            self.__display.on_image_received(image_cpy)
            self.__display.update()

            # Increase frame counter
            self.__frame_counter += 1
        except ids_peak.Exception as e:
            self.__error_counter += 1
            print("Exception: " + str(e))

        # Update counters
        self.update_counters()

    def impro(self,img,width,height):
        scale_percent = 20 # percent of original size
        width2 = int(width * scale_percent / 100)
        height2 = int(height * scale_percent / 100)
        dim = (width2, height2)
        # print(str(width)+","+str(height)+" "+str(width2))
        # ...resize the image by a half
        frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) #cv2.resize(frame_origin,(0,0),fx=0.5, fy=0.5)
        # frame_origin = cv2.rotate(frame_origin,cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    #---------------------------------------------------------------------------------------------------------------------------------------
        #Include image data processing here
        # image_blur = cv2.GaussianBlur(frame, (5, 5), 0)#cv2.medianBlur(frame,3)
        # image_blur_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    

    
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Filters
        color_lower_bound = np.array([30, 150, 130])
        color_upper_bound = np.array([45, 256, 256])
        sobelxy = cv2.Sobel(src=image_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) # Combined X
        lowpass = ndimage.gaussian_filter(image_gray, 5)
        gauss_highpass = image_gray - lowpass
        # mask_color = cv2.inRange(image_blur_hsv, color_lower_bound, color_upper_bound)
    #---------------------------------------------------------------------------------------------------------------------------------------

        # ...and finally display it
        # ret = comm.Read('VB.Position')
        # print(ret.TagName, ret.Value, ret.Status)
    
        # if ret.Value :
        
        #  i +=1qqqqq
        # alpha = 1.5 # Contrast control (1.0-3.0)
        # beta = 0 # Brightness control (0-100)

        # adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
        #contrast
        # hist,bins = np.histogram(image_gray.flatten(),256,[0,256])
        # cdf = hist.cumsum()
        # cdf_normalized = cdf * float(hist.max()) / cdf.max()
        # cdf_m = np.ma.masked_equal(cdf,0)
        # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        # cdf = np.ma.filled(cdf_m,0).astype('uint8')
        # img2 = cdf[image_gray]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        # cl1 = image_gray
        cl1 = clahe.apply(image_gray)
        # edges = cv2.Canny(cl1,10,200)
        # equ = cv2.equalizeHist(image_gray)
        # res = np.hstack((image_gray,equ)) #stacking images side-by-side
        ddept=cv2.CV_32F
        x = cv2.Sobel(cl1, ddept, 1,0, ksize=3, scale=1)
        y = cv2.Sobel(cl1, ddept, 0,1, ksize=3, scale=1)
        absx= cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        edge = cv2.addWeighted(absx, 4, absy, 4,0)
        
    
        crop_img = edge[int(self.yy-self.h/2):int(self.yy+self.h/2), int(self.xx-self.w/2):int(self.xx+self.w/2)]
        edge_re = cv2.resize( edge, dim, interpolation = cv2.INTER_AREA)
        im_resized_norm = cv2.resize(crop_img, (500, 500), interpolation=cv2.INTER_LINEAR)
        image_color = cv2.cvtColor(im_resized_norm, cv2.COLOR_GRAY2RGB)
        image_color_crop = cv2.cvtColor(edge_re, cv2.COLOR_GRAY2RGB)
        
        # lastTime = time.time()
        # timecount = lastTime - startTime
        # if timecount >3:
        #     print(timecount)
        #     startTime = time.time()
        #     cv2.imwrite('D:/Work2021/test/image_data2/52821_Base2/'+str(filename)+'.jpg', edge)
        #     filename =filename +1
        #     comm.Write('VB.Velocity',1000)
        #     comm.Write('VB.Position',100)
        #     comm.Write('VB.Start',True)

        cv2.drawMarker(image_color, (self.markx,self.marky), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=1)
        cv2.drawMarker(image_color_crop, (self.markx1,self.marky1), color=(255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.imshow("crop", image_color)
        cv2.imshow("full", image_color_crop)
        cv2.setMouseCallback('full',self.draw_circle)
        cv2.setMouseCallback('crop',self.draw_marker)
   

    @Slot(str)
    def on_aboutqt_link_activated(self, link):
        if link == "#aboutQt":
            QMessageBox.aboutQt(self, "About Qt")
