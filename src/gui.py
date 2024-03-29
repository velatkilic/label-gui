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
        MainWindow.resize(1015, 822)
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
        self.button_auto_detect = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_auto_detect.setFont(font)
        self.button_auto_detect.setObjectName("button_auto_detect")
        self.verticalLayout.addWidget(self.button_auto_detect)
        self.button_embedding = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_embedding.setFont(font)
        self.button_embedding.setObjectName("button_embedding")
        self.verticalLayout.addWidget(self.button_embedding)
        self.button_query_prev = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_query_prev.setFont(font)
        self.button_query_prev.setObjectName("button_query_prev")
        self.verticalLayout.addWidget(self.button_query_prev)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.radio_annot_on = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radio_annot_on.setFont(font)
        self.radio_annot_on.setChecked(False)
        self.radio_annot_on.setObjectName("radio_annot_on")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radio_annot_on)
        self.horizontalLayout_3.addWidget(self.radio_annot_on)
        self.radio_annot_off = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radio_annot_off.setFont(font)
        self.radio_annot_off.setChecked(True)
        self.radio_annot_off.setObjectName("radio_annot_off")
        self.buttonGroup.addButton(self.radio_annot_off)
        self.horizontalLayout_3.addWidget(self.radio_annot_off)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_mask_scale = QtWidgets.QHBoxLayout()
        self.horizontalLayout_mask_scale.setObjectName("horizontalLayout_mask_scale")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_mask_scale.addWidget(self.label_3)
        self.spinBox_mask_scale = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_mask_scale.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_mask_scale.setFont(font)
        self.spinBox_mask_scale.setMinimum(0)
        self.spinBox_mask_scale.setMaximum(2)
        self.spinBox_mask_scale.setProperty("value", 1)
        self.spinBox_mask_scale.setObjectName("spinBox_mask_scale")
        self.horizontalLayout_mask_scale.addWidget(self.spinBox_mask_scale)
        self.verticalLayout_4.addLayout(self.horizontalLayout_mask_scale)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_10.addWidget(self.label_4)
        self.radio_mask_all = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radio_mask_all.setFont(font)
        self.radio_mask_all.setObjectName("radio_mask_all")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radio_mask_all)
        self.horizontalLayout_10.addWidget(self.radio_mask_all)
        self.radio_mask_last = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.radio_mask_last.setFont(font)
        self.radio_mask_last.setChecked(True)
        self.radio_mask_last.setObjectName("radio_mask_last")
        self.buttonGroup_2.addButton(self.radio_mask_last)
        self.horizontalLayout_10.addWidget(self.radio_mask_last)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.annot_list = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.annot_list.sizePolicy().hasHeightForWidth())
        self.annot_list.setSizePolicy(sizePolicy)
        self.annot_list.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed)
        self.annot_list.setObjectName("annot_list")
        self.verticalLayout.addWidget(self.annot_list)
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
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
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
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.spinBox_frame_id = QtWidgets.QSpinBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_frame_id.setFont(font)
        self.spinBox_frame_id.setMaximum(999999999)
        self.spinBox_frame_id.setObjectName("spinBox_frame_id")
        self.horizontalLayout_5.addWidget(self.spinBox_frame_id)
        self.label_frame_count = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_frame_count.setFont(font)
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1015, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuDnn = QtWidgets.QMenu(self.menubar)
        self.menuDnn.setObjectName("menuDnn")
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
        self.action_save_annot = QtWidgets.QAction(MainWindow)
        self.action_save_annot.setObjectName("action_save_annot")
        self.action_toggle_side_menu = QtWidgets.QAction(MainWindow)
        self.action_toggle_side_menu.setCheckable(True)
        self.action_toggle_side_menu.setObjectName("action_toggle_side_menu")
        self.actionROI = QtWidgets.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.action_roi_settings = QtWidgets.QAction(MainWindow)
        self.action_roi_settings.setObjectName("action_roi_settings")
        self.action_load_video = QtWidgets.QAction(MainWindow)
        self.action_load_video.setObjectName("action_load_video")
        self.action_load_embeddings = QtWidgets.QAction(MainWindow)
        self.action_load_embeddings.setObjectName("action_load_embeddings")
        self.action_save_embeddings = QtWidgets.QAction(MainWindow)
        self.action_save_embeddings.setObjectName("action_save_embeddings")
        self.action_load_SAM = QtWidgets.QAction(MainWindow)
        self.action_load_SAM.setObjectName("action_load_SAM")
        self.menuFile.addAction(self.action_load_video)
        self.menuFile.addAction(self.action_load_images)
        self.menuFile.addAction(self.action_load_annot)
        self.menuFile.addAction(self.action_load_embeddings)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.action_save_annot)
        self.menuFile.addAction(self.action_save_embeddings)
        self.menuDnn.addAction(self.action_load_SAM)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuDnn.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.radio_annot_on, self.radio_annot_off)
        MainWindow.setTabOrder(self.radio_annot_off, self.spinBox_mask_scale)
        MainWindow.setTabOrder(self.spinBox_mask_scale, self.class_label)
        MainWindow.setTabOrder(self.class_label, self.button_add_class_label)
        MainWindow.setTabOrder(self.button_add_class_label, self.label_list)
        MainWindow.setTabOrder(self.label_list, self.spinBox_frame_id)
        MainWindow.setTabOrder(self.spinBox_frame_id, self.button_prev)
        MainWindow.setTabOrder(self.button_prev, self.button_auto_detect)
        MainWindow.setTabOrder(self.button_auto_detect, self.button_next)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_auto_detect.setText(_translate("MainWindow", "Auto-Detect"))
        self.button_embedding.setText(_translate("MainWindow", "Pre-Compute Embeddings"))
        self.button_query_prev.setText(_translate("MainWindow", "Query Prev Frame Detections"))
        self.label.setText(_translate("MainWindow", "Annotate"))
        self.radio_annot_on.setText(_translate("MainWindow", "On"))
        self.radio_annot_off.setText(_translate("MainWindow", "Off"))
        self.label_3.setText(_translate("MainWindow", "Mask Scale"))
        self.label_4.setText(_translate("MainWindow", "Show Mask"))
        self.radio_mask_all.setText(_translate("MainWindow", "All"))
        self.radio_mask_last.setText(_translate("MainWindow", "Last"))
        self.class_label.setText(_translate("MainWindow", "class label"))
        self.button_add_class_label.setText(_translate("MainWindow", "Add"))
        self.label_2.setText(_translate("MainWindow", "Frame ID"))
        self.label_frame_count.setText(_translate("MainWindow", "/0"))
        self.button_prev.setText(_translate("MainWindow", "Prev"))
        self.button_next.setText(_translate("MainWindow", "Next"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuDnn.setTitle(_translate("MainWindow", "Dnn"))
        self.action_load_images.setText(_translate("MainWindow", "Load Image Folder"))
        self.action_load_images.setStatusTip(_translate("MainWindow", "Load Images. Navigate to a video or images folder"))
        self.action_load_annot.setText(_translate("MainWindow", "Load Annotations"))
        self.action_load_annot.setStatusTip(_translate("MainWindow", "Load annotations: bounding boxes, segmentation mask"))
        self.action_load_masks.setText(_translate("MainWindow", "Load Masks"))
        self.action_load_masks.setStatusTip(_translate("MainWindow", "Load segmentation masks"))
        self.action_save_annot.setText(_translate("MainWindow", "Save Annotations"))
        self.action_save_annot.setStatusTip(_translate("MainWindow", "Save annotations"))
        self.action_toggle_side_menu.setText(_translate("MainWindow", "Side Menu"))
        self.action_toggle_side_menu.setStatusTip(_translate("MainWindow", "Hide/Unhide class label side menu"))
        self.actionROI.setText(_translate("MainWindow", "ROI"))
        self.action_roi_settings.setText(_translate("MainWindow", "ROI"))
        self.action_load_video.setText(_translate("MainWindow", "Load Video File"))
        self.action_load_embeddings.setText(_translate("MainWindow", "Load Embeddings"))
        self.action_save_embeddings.setText(_translate("MainWindow", "Save Embeddings"))
        self.action_load_SAM.setText(_translate("MainWindow", "Load SAM"))
from pyqtgraph import GraphicsLayoutWidget
