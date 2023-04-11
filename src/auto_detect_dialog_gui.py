# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'auto_detect_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(240, 504)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.spinBox_points_per_side = QtWidgets.QSpinBox(Dialog)
        self.spinBox_points_per_side.setMinimum(1)
        self.spinBox_points_per_side.setMaximum(100000)
        self.spinBox_points_per_side.setProperty("value", 32)
        self.spinBox_points_per_side.setObjectName("spinBox_points_per_side")
        self.horizontalLayout.addWidget(self.spinBox_points_per_side)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.spinBox_quality_threshold = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_quality_threshold.setMaximum(1.0)
        self.spinBox_quality_threshold.setSingleStep(0.1)
        self.spinBox_quality_threshold.setProperty("value", 0.88)
        self.spinBox_quality_threshold.setObjectName("spinBox_quality_threshold")
        self.horizontalLayout_2.addWidget(self.spinBox_quality_threshold)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.spinBox_stability_threshold = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_stability_threshold.setMaximum(1.0)
        self.spinBox_stability_threshold.setSingleStep(0.1)
        self.spinBox_stability_threshold.setProperty("value", 0.95)
        self.spinBox_stability_threshold.setObjectName("spinBox_stability_threshold")
        self.horizontalLayout_3.addWidget(self.spinBox_stability_threshold)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_8.addWidget(self.label_8)
        self.spinBox_nms_threshold = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_nms_threshold.setMaximum(1.0)
        self.spinBox_nms_threshold.setSingleStep(0.1)
        self.spinBox_nms_threshold.setProperty("value", 0.7)
        self.spinBox_nms_threshold.setObjectName("spinBox_nms_threshold")
        self.horizontalLayout_8.addWidget(self.spinBox_nms_threshold)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.spinBox_crop_n_layers = QtWidgets.QSpinBox(Dialog)
        self.spinBox_crop_n_layers.setProperty("value", 0)
        self.spinBox_crop_n_layers.setObjectName("spinBox_crop_n_layers")
        self.horizontalLayout_4.addWidget(self.spinBox_crop_n_layers)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_9.addWidget(self.label_9)
        self.spinBox_crop_nms_threshold = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_crop_nms_threshold.setMaximum(1.0)
        self.spinBox_crop_nms_threshold.setSingleStep(0.1)
        self.spinBox_crop_nms_threshold.setProperty("value", 0.7)
        self.spinBox_crop_nms_threshold.setObjectName("spinBox_crop_nms_threshold")
        self.horizontalLayout_9.addWidget(self.spinBox_crop_nms_threshold)
        self.verticalLayout.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_10.addWidget(self.label_10)
        self.spinBox_crop_overlap_ratio = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_crop_overlap_ratio.setMaximum(1.0)
        self.spinBox_crop_overlap_ratio.setSingleStep(0.1)
        self.spinBox_crop_overlap_ratio.setProperty("value", 0.7)
        self.spinBox_crop_overlap_ratio.setObjectName("spinBox_crop_overlap_ratio")
        self.horizontalLayout_10.addWidget(self.spinBox_crop_overlap_ratio)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_5.addWidget(self.label_5)
        self.spinBox_max_mask_region = QtWidgets.QSpinBox(Dialog)
        self.spinBox_max_mask_region.setMaximum(999999999)
        self.spinBox_max_mask_region.setSingleStep(1000)
        self.spinBox_max_mask_region.setProperty("value", 10000)
        self.spinBox_max_mask_region.setObjectName("spinBox_max_mask_region")
        self.horizontalLayout_5.addWidget(self.spinBox_max_mask_region)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        self.spinBox_min_mask_region = QtWidgets.QSpinBox(Dialog)
        self.spinBox_min_mask_region.setMaximum(99999)
        self.spinBox_min_mask_region.setObjectName("spinBox_min_mask_region")
        self.horizontalLayout_6.addWidget(self.spinBox_min_mask_region)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        self.spinBox_points_per_batch = QtWidgets.QSpinBox(Dialog)
        self.spinBox_points_per_batch.setMinimum(1)
        self.spinBox_points_per_batch.setMaximum(100000)
        self.spinBox_points_per_batch.setProperty("value", 64)
        self.spinBox_points_per_batch.setObjectName("spinBox_points_per_batch")
        self.horizontalLayout_7.addWidget(self.spinBox_points_per_batch)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.spinBox_points_per_side, self.spinBox_quality_threshold)
        Dialog.setTabOrder(self.spinBox_quality_threshold, self.spinBox_stability_threshold)
        Dialog.setTabOrder(self.spinBox_stability_threshold, self.spinBox_nms_threshold)
        Dialog.setTabOrder(self.spinBox_nms_threshold, self.spinBox_crop_n_layers)
        Dialog.setTabOrder(self.spinBox_crop_n_layers, self.spinBox_crop_nms_threshold)
        Dialog.setTabOrder(self.spinBox_crop_nms_threshold, self.spinBox_crop_overlap_ratio)
        Dialog.setTabOrder(self.spinBox_crop_overlap_ratio, self.spinBox_max_mask_region)
        Dialog.setTabOrder(self.spinBox_max_mask_region, self.spinBox_min_mask_region)
        Dialog.setTabOrder(self.spinBox_min_mask_region, self.spinBox_points_per_batch)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setStatusTip(_translate("Dialog", "Number of points to be sampled"))
        self.label.setText(_translate("Dialog", "Points per side"))
        self.spinBox_points_per_side.setStatusTip(_translate("Dialog", "Number of points to be sampled"))
        self.label_2.setStatusTip(_translate("Dialog", "Filtering using model\'s predicted mask quality"))
        self.label_2.setText(_translate("Dialog", "Quality Threshold"))
        self.spinBox_quality_threshold.setStatusTip(_translate("Dialog", "Filtering using model\'s predicted mask quality"))
        self.label_3.setStatusTip(_translate("Dialog", "Mask stability threshold"))
        self.label_3.setText(_translate("Dialog", "Stability Threshold"))
        self.spinBox_stability_threshold.setStatusTip(_translate("Dialog", "Mask stability threshold"))
        self.label_8.setStatusTip(_translate("Dialog", "Intersection over union threshold used for NMS"))
        self.label_8.setText(_translate("Dialog", "NMS Threshold"))
        self.spinBox_nms_threshold.setStatusTip(_translate("Dialog", "Intersection over union threshold used for NMS"))
        self.label_4.setStatusTip(_translate("Dialog", "Mask prediction will be run again on crops of the image"))
        self.label_4.setText(_translate("Dialog", "Crop N Layers"))
        self.spinBox_crop_n_layers.setStatusTip(_translate("Dialog", "Mask prediction will be run again on crops of the image"))
        self.label_9.setStatusTip(_translate("Dialog", "Intersection over union threshold used for NMS to combine crops"))
        self.label_9.setText(_translate("Dialog", "Crop NMS Threshold"))
        self.spinBox_crop_nms_threshold.setStatusTip(_translate("Dialog", "Intersection over union threshold used for NMS to combine crops"))
        self.label_10.setStatusTip(_translate("Dialog", "crops will overlap by this fraction"))
        self.label_10.setText(_translate("Dialog", "Crop Overlap Ratio"))
        self.spinBox_crop_overlap_ratio.setStatusTip(_translate("Dialog", "crops will overlap by this fraction"))
        self.label_5.setStatusTip(_translate("Dialog", "Threshold masks larger than this"))
        self.label_5.setText(_translate("Dialog", "Max Mask Size"))
        self.spinBox_max_mask_region.setStatusTip(_translate("Dialog", "Threshold masks larger than this"))
        self.label_6.setStatusTip(_translate("Dialog", "Use this to remove disconnected regions and holes in masks unless set to 0."))
        self.label_6.setText(_translate("Dialog", "Min Mask Region"))
        self.spinBox_min_mask_region.setStatusTip(_translate("Dialog", "Use this to remove disconnected regions and holes in masks unless set to 0."))
        self.label_7.setStatusTip(_translate("Dialog", "number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory."))
        self.label_7.setText(_translate("Dialog", "Points per batch"))
        self.spinBox_points_per_batch.setStatusTip(_translate("Dialog", "number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory."))
