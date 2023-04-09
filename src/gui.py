# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(796, 591)
        MainWindow.setAccessibleName("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.img_view = GraphicsLayoutWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_view.sizePolicy().hasHeightForWidth())
        self.img_view.setSizePolicy(sizePolicy)
        self.img_view.setMinimumSize(QtCore.QSize(0, 0))
        self.img_view.setObjectName("img_view")
        self.horizontalLayout_2.addWidget(self.img_view)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.radio_mask_on = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_mask_on.setChecked(False)
        self.radio_mask_on.setObjectName("radio_mask_on")
        self.horizontalLayout_3.addWidget(self.radio_mask_on)
        self.radio_mask_off = QtWidgets.QRadioButton(self.centralwidget)
        self.radio_mask_off.setChecked(True)
        self.radio_mask_off.setObjectName("radio_mask_off")
        self.horizontalLayout_3.addWidget(self.radio_mask_off)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_mask_scale = QtWidgets.QHBoxLayout()
        self.horizontalLayout_mask_scale.setObjectName("horizontalLayout_mask_scale")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_mask_scale.addWidget(self.label_3)
        self.spinBox_mask_scale = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_mask_scale.setMinimum(0)
        self.spinBox_mask_scale.setMaximum(2)
        self.spinBox_mask_scale.setProperty("value", 0)
        self.spinBox_mask_scale.setObjectName("spinBox_mask_scale")
        self.horizontalLayout_mask_scale.addWidget(self.spinBox_mask_scale)
        self.verticalLayout_4.addLayout(self.horizontalLayout_mask_scale)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.class_label = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.class_label.sizePolicy().hasHeightForWidth())
        self.class_label.setSizePolicy(sizePolicy)
        self.class_label.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.class_label.setFont(font)
        self.class_label.setObjectName("class_label")
        self.horizontalLayout.addWidget(self.class_label)
        self.button_add_class_label = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_add_class_label.sizePolicy().hasHeightForWidth())
        self.button_add_class_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_add_class_label.setFont(font)
        self.button_add_class_label.setObjectName("button_add_class_label")
        self.horizontalLayout.addWidget(self.button_add_class_label)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_list = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_list.sizePolicy().hasHeightForWidth())
        self.label_list.setSizePolicy(sizePolicy)
        self.label_list.setObjectName("label_list")
        self.verticalLayout.addWidget(self.label_list)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.spinBox_frame_id = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_frame_id.setMaximum(999999999)
        self.spinBox_frame_id.setObjectName("spinBox_frame_id")
        self.horizontalLayout_5.addWidget(self.spinBox_frame_id)
        self.label_frame_count = QtWidgets.QLabel(self.centralwidget)
        self.label_frame_count.setObjectName("label_frame_count")
        self.horizontalLayout_5.addWidget(self.label_frame_count)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.button_prev = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_prev.setFont(font)
        self.button_prev.setObjectName("button_prev")
        self.verticalLayout_2.addWidget(self.button_prev)
        self.button_next = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_next.setFont(font)
        self.button_next.setObjectName("button_next")
        self.verticalLayout_2.addWidget(self.button_next)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 796, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_load_images = QtWidgets.QAction(MainWindow)
        self.action_load_images.setObjectName("action_load_images")
        self.action_load_annot = QtWidgets.QAction(MainWindow)
        self.action_load_annot.setObjectName("action_load_annot")
        self.action_load_masks = QtWidgets.QAction(MainWindow)
        self.action_load_masks.setObjectName("action_load_masks")
        self.action_save = QtWidgets.QAction(MainWindow)
        self.action_save.setObjectName("action_save")
        self.action_toggle_side_menu = QtWidgets.QAction(MainWindow)
        self.action_toggle_side_menu.setCheckable(True)
        self.action_toggle_side_menu.setObjectName("action_toggle_side_menu")
        self.actionROI = QtWidgets.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.action_roi_settings = QtWidgets.QAction(MainWindow)
        self.action_roi_settings.setObjectName("action_roi_settings")
        self.action_load_video = QtWidgets.QAction(MainWindow)
        self.action_load_video.setObjectName("action_load_video")
        self.menuFile.addAction(self.action_load_video)
        self.menuFile.addAction(self.action_load_images)
        self.menuFile.addAction(self.action_load_annot)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_save)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radio_mask_on.setText(_translate("MainWindow", "Mask On"))
        self.radio_mask_off.setText(_translate("MainWindow", "Mask Off"))
        self.label_3.setText(_translate("MainWindow", "Mask Scale"))
        self.class_label.setText(_translate("MainWindow", "class label"))
        self.button_add_class_label.setText(_translate("MainWindow", "Add"))
        self.label_2.setText(_translate("MainWindow", "Frame ID"))
        self.label_frame_count.setText(_translate("MainWindow", "/0"))
        self.button_prev.setText(_translate("MainWindow", "Prev"))
        self.button_next.setText(_translate("MainWindow", "Next"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.action_load_images.setText(_translate("MainWindow", "Load Image Folder"))
        self.action_load_images.setStatusTip(_translate("MainWindow", "Load Images. Navigate to a video or images folder"))
        self.action_load_annot.setText(_translate("MainWindow", "Load Annotations"))
        self.action_load_annot.setStatusTip(_translate("MainWindow", "Load annotations: bounding boxes, segmentation mask"))
        self.action_load_masks.setText(_translate("MainWindow", "Load Masks"))
        self.action_load_masks.setStatusTip(_translate("MainWindow", "Load segmentation masks"))
        self.action_save.setText(_translate("MainWindow", "Save"))
        self.action_save.setStatusTip(_translate("MainWindow", "Save annotations"))
        self.action_toggle_side_menu.setText(_translate("MainWindow", "Side Menu"))
        self.action_toggle_side_menu.setStatusTip(_translate("MainWindow", "Hide/Unhide class label side menu"))
        self.actionROI.setText(_translate("MainWindow", "ROI"))
        self.action_roi_settings.setText(_translate("MainWindow", "ROI"))
        self.action_load_video.setText(_translate("MainWindow", "Load Video File"))
from pyqtgraph import GraphicsLayoutWidget
