# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/main_window.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(963, 643)
        Form.setStyleSheet("background-color: rgb(254, 205, 166);")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_5 = QtWidgets.QGroupBox(Form)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.radio_sh_cheb = QtWidgets.QRadioButton(self.groupBox_3)
        self.radio_sh_cheb.setChecked(True)
        self.radio_sh_cheb.setVisible(False)
        self.radio_sh_cheb.setObjectName("radio_sh_cheb")
        self.verticalLayout.addWidget(self.radio_sh_cheb)
        self.radio_cheb = QtWidgets.QRadioButton(self.groupBox_3)
        self.radio_cheb.setObjectName("radio_cheb")
        self.radio_cheb.setChecked(True)
        self.verticalLayout.addWidget(self.radio_cheb)
        # self.radio_sh_cheb_2 = QtWidgets.QRadioButton(self.groupBox_3)
        # self.radio_sh_cheb_2.setObjectName("radio_sh_cheb_2")
        # self.verticalLayout.addWidget(self.radio_sh_cheb_2)
        self.custom_check = QtWidgets.QCheckBox(self.groupBox_3)
        self.custom_check.setChecked(True)
        self.custom_check.setObjectName("custom_check")
        self.verticalLayout.addWidget(self.custom_check)
        self.horizontalLayout.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_4.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 1)
        self.x1_dim = QtWidgets.QSpinBox(self.groupBox_2)
        self.x1_dim.setEnabled(True)
        self.x1_dim.setProperty("value", 4)
        self.x1_dim.setObjectName("x1_dim")
        self.gridLayout_3.addWidget(self.x1_dim, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.y_dim = QtWidgets.QSpinBox(self.groupBox_2)
        self.y_dim.setEnabled(True)
        self.y_dim.setProperty("value", 3)
        self.y_dim.setObjectName("y_dim")
        self.gridLayout_3.addWidget(self.y_dim, 3, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_3.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_5.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 3, 0, 1, 1)
        self.x2_dim = QtWidgets.QSpinBox(self.groupBox_2)
        self.x2_dim.setEnabled(True)
        self.x2_dim.setProperty("value", 2)
        self.x2_dim.setObjectName("x2_dim")
        self.gridLayout_3.addWidget(self.x2_dim, 1, 1, 1, 1)
        self.x3_dim = QtWidgets.QSpinBox(self.groupBox_2)
        self.x3_dim.setEnabled(True)
        self.x3_dim.setProperty("value", 3)
        self.x3_dim.setObjectName("x3_dim")
        self.gridLayout_3.addWidget(self.x3_dim, 2, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBox_2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_6 = QtWidgets.QLabel(self.groupBox_4)
        self.label_6.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_6.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 0, 0, 1, 1)
        self.x3_deg = QtWidgets.QSpinBox(self.groupBox_4)
        self.x3_deg.setProperty("value", 1)
        self.x3_deg.setObjectName("x3_deg")
        self.gridLayout_4.addWidget(self.x3_deg, 2, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_7.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 2, 0, 1, 1)
        self.x2_deg = QtWidgets.QSpinBox(self.groupBox_4)
        self.x2_deg.setProperty("value", 1)
        self.x2_deg.setObjectName("x2_deg")
        self.gridLayout_4.addWidget(self.x2_deg, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_4)
        self.label_8.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_8.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_4.addWidget(self.label_8, 1, 0, 1, 1)
        self.x1_deg = QtWidgets.QSpinBox(self.groupBox_4)
        self.x1_deg.setProperty("value", 1)
        self.x1_deg.setObjectName("x1_deg")
        self.gridLayout_4.addWidget(self.x1_deg, 0, 1, 1, 1)
        self.bruteforce_btn = QtWidgets.QPushButton(self.groupBox_4)
        self.bruteforce_btn.setEnabled(True)
        self.bruteforce_btn.setVisible(False)
        self.bruteforce_btn.setObjectName("bruteforce_btn")
        self.bruteforce_btn.setStyleSheet("background-color: rgb(150, 150, 150)")
        self.gridLayout_4.addWidget(self.bruteforce_btn, 3, 0, 1, 2)
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 2)
        self.horizontalLayout.addWidget(self.groupBox_4)
        self.horizontalLayout_8.addWidget(self.groupBox_5)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_8)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_7 = QtWidgets.QGroupBox(Form)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.groupBox_7)
        self.label_9.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.weights_box = QtWidgets.QComboBox(self.groupBox_7)
        self.weights_box.setEnabled(True)
        self.weights_box.setObjectName("weights_box")
        self.weights_box.addItem("")
        self.weights_box.addItem("")
        self.horizontalLayout_3.addWidget(self.weights_box)
        self.lambda_check = QtWidgets.QCheckBox(self.groupBox_7)
        self.lambda_check.setEnabled(True)
        self.lambda_check.setObjectName("lambda_check")
        self.horizontalLayout_3.addWidget(self.lambda_check)
        self.verticalLayout_2.addWidget(self.groupBox_7)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.line_output = QtWidgets.QLineEdit(self.groupBox)
        self.line_output.setText("")
        self.line_output.setObjectName("line_output")
        self.gridLayout_5.addWidget(self.line_output, 2, 0, 1, 1)
        self.select_output = QtWidgets.QToolButton(self.groupBox)
        self.select_output.setObjectName("select_output")
        self.gridLayout_5.addWidget(self.select_output, 2, 1, 1, 1)
        self.line_input = QtWidgets.QLineEdit(self.groupBox)
        self.line_input.setText("")
        self.line_input.setObjectName("line_input")
        self.gridLayout_5.addWidget(self.line_input, 1, 0, 1, 1)
        self.select_input = QtWidgets.QToolButton(self.groupBox)
        self.select_input.setObjectName("select_input")
        self.gridLayout_5.addWidget(self.select_input, 1, 1, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_9, 0, 0, 2, 2)
        self.tablewidget = QtWidgets.QTableWidget(Form)
        self.tablewidget.setMaximumSize(QtCore.QSize(9000, 16777215))
        self.tablewidget.setRowCount(10)
        self.tablewidget.setColumnCount(8)
        self.tablewidget.setObjectName("tablewidget")
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tablewidget.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        item.setFont(font)
        self.tablewidget.setItem(0, 0, item)
        self.gridLayout_2.addWidget(self.tablewidget, 3, 0, 1, 2)
        self.Current = QtWidgets.QGroupBox(Form)
        self.Current.setObjectName("Current")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.Current)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_11 = QtWidgets.QLabel(self.Current)
        self.label_11.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_11.setObjectName("label_11")
        self.gridLayout_6.addWidget(self.label_11, 0, 0, 2, 1)
        self.lbl_time = QtWidgets.QLabel(self.Current)
        self.lbl_time.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.lbl_time.setText("")
        self.lbl_time.setObjectName("lbl_time")
        self.gridLayout_6.addWidget(self.lbl_time, 0, 1, 2, 1)
        self.label_12 = QtWidgets.QLabel(self.Current)
        self.label_12.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 2, 1, 1)
        self.lbl_y1 = QtWidgets.QLabel(self.Current)
        self.lbl_y1.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.lbl_y1.setText("")
        self.lbl_y1.setObjectName("lbl_y1")
        self.gridLayout_6.addWidget(self.lbl_y1, 0, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.Current)
        self.label_13.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_13.setObjectName("label_13")
        self.gridLayout_6.addWidget(self.label_13, 1, 2, 1, 1)
        self.lbl_y2 = QtWidgets.QLabel(self.Current)
        self.lbl_y2.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.lbl_y2.setText("")
        self.lbl_y2.setObjectName("lbl_y2")
        self.gridLayout_6.addWidget(self.lbl_y2, 1, 3, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.Current)
        self.label_15.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_15.setObjectName("label_15")
        self.gridLayout_6.addWidget(self.label_15, 2, 0, 1, 1)
        self.lbl_rmr = QtWidgets.QLabel(self.Current)
        self.lbl_rmr.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.lbl_rmr.setText("")
        self.lbl_rmr.setObjectName("lbl_rmr")
        self.gridLayout_6.addWidget(self.lbl_rmr, 2, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.Current)
        self.label_14.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_14.setObjectName("label_14")
        self.gridLayout_6.addWidget(self.label_14, 2, 2, 1, 1)
        self.lbl_y3 = QtWidgets.QLabel(self.Current)
        self.lbl_y3.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.lbl_y3.setText("")

        self.lbl_y3.setObjectName("lbl_y3")
        self.gridLayout_6.addWidget(self.lbl_y3, 2, 3, 1, 1)
        self.label_11.raise_()
        self.lbl_time.raise_()
        self.label_15.raise_()
        self.lbl_rmr.raise_()
        self.label_12.raise_()
        self.lbl_y1.raise_()
        self.label_13.raise_()
        self.lbl_y2.raise_()
        self.label_14.raise_()
        self.lbl_y3.raise_()
        self.gridLayout_2.addWidget(self.Current, 2, 0, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(Form)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout.setObjectName("gridLayout")
        self.label_10 = QtWidgets.QLabel(self.groupBox_6)
        self.label_10.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 3, 0, 1, 1)
        self.predictBox = QtWidgets.QSpinBox(self.groupBox_6)
        self.predictBox.setProperty("value", 10)
        self.predictBox.setObjectName("predictBox")
        self.gridLayout.addWidget(self.predictBox, 3, 2, 1, 1)
        self.exec_button = QtWidgets.QPushButton(self.groupBox_6)
        self.exec_button.setStyleSheet("background-color: rgb(166, 207, 152); border-radius: 5px; padding:2px; font-size:15px;")
        self.exec_button.setDefault(True)
        self.exec_button.setFlat(False)
        # self.exec_button.setVisible(False)
        self.exec_button.setObjectName("exec_button")
        self.gridLayout.addWidget(self.exec_button, 2, 3, 3, 1)
        self.sample_spin = QtWidgets.QSpinBox(self.groupBox_6)
        self.sample_spin.setMaximum(9990)
        self.sample_spin.setProperty("value", 50)
        self.sample_spin.setObjectName("sample_spin")
        self.gridLayout.addWidget(self.sample_spin, 1, 2, 2, 1)
        self.label = QtWidgets.QLabel(self.groupBox_6)
        self.label.setStyleSheet("background-color: rgb(230, 249, 255); border-radius: 5px; padding:2px")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 2, 1)
        self.gridLayout_2.addWidget(self.groupBox_6, 2, 1, 1, 1)

        self.retranslateUi(Form)
        self.weights_box.setCurrentIndex(1)
        self.exec_button.clicked.connect(Form.exec_clicked)
        self.select_input.clicked.connect(Form.input_clicked)
        self.select_output.clicked.connect(Form.output_clicked)
        self.x1_dim.valueChanged["int"].connect(Form.dimension_modified)
        self.x2_dim.valueChanged["int"].connect(Form.dimension_modified)
        self.x3_dim.valueChanged["int"].connect(Form.dimension_modified)
        self.x1_deg.valueChanged["int"].connect(Form.degree_modified)
        self.x2_deg.valueChanged["int"].connect(Form.degree_modified)
        self.x3_deg.valueChanged["int"].connect(Form.degree_modified)
        self.radio_sh_cheb.toggled["bool"].connect(Form.type_modified)
        # self.radio_sh_cheb_2.toggled['bool'].connect(Form.type_modified)
        self.radio_cheb.toggled["bool"].connect(Form.type_modified)
        Form.output_changed["QString"].connect(self.line_output.setText)
        Form.input_changed["QString"].connect(self.line_input.setText)
        self.line_input.textChanged["QString"].connect(Form.input_modified)
        self.line_output.textChanged["QString"].connect(Form.output_modified)
        self.lambda_check.toggled["bool"].connect(Form.lambda_calc_method_changed)
        self.weights_box.currentIndexChanged["QString"].connect(Form.weights_modified)
        self.y_dim.valueChanged["int"].connect(Form.dimension_modified)
        self.bruteforce_btn.clicked.connect(Form.bruteforce_called)
        self.custom_check.toggled["bool"].connect(Form.structure_changed)
        self.sample_spin.valueChanged["int"].connect(Form.samples_modified)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.radio_sh_cheb, self.radio_cheb)
        # Form.setTabOrder(self.radio_cheb, self.radio_sh_cheb_2)
        # Form.setTabOrder(self.radio_sh_cheb_2, self.x1_deg)
        Form.setTabOrder(self.x1_deg, self.x2_deg)
        Form.setTabOrder(self.x2_deg, self.x3_deg)
        Form.setTabOrder(self.x3_deg, self.x1_dim)
        Form.setTabOrder(self.x1_dim, self.x2_dim)
        Form.setTabOrder(self.x2_dim, self.x3_dim)
        Form.setTabOrder(self.x3_dim, self.y_dim)
        Form.setTabOrder(self.y_dim, self.line_input)
        Form.setTabOrder(self.line_input, self.select_input)
        Form.setTabOrder(self.select_input, self.line_output)
        Form.setTabOrder(self.line_output, self.select_output)
        
        widget_to_swap = self.groupBox_5
        widgets_to_insert = [self.groupBox, self.groupBox_7]

        # Remove the widget_to_swap from its current layout
        self.horizontalLayout_8.removeWidget(widget_to_swap)

        # Create a new layout to insert the widgets_to_insert
        new_layout = QtWidgets.QHBoxLayout()

        # Insert the widgets_to_insert into the new layout
        for widget in widgets_to_insert:
            new_layout.addWidget(widget)

        # Insert the widget_to_swap into the new layout
        new_layout.addWidget(widget_to_swap)

        # Add the new layout to the existing layout
        self.horizontalLayout_8.addLayout(new_layout)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form1"))
        #
        self.groupBox_5.setTitle(_translate("Form", "Поліноми"))
        self.groupBox_5.setStyleSheet("font-weight: bold; font-size: 14")
        self.groupBox_3.setTitle(_translate("Form", "Типи"))
        self.radio_sh_cheb.setText(_translate("Form", "Комбінований"))
        self.radio_cheb.setText(_translate("Form", "поліноми Чебишева"))
        # self.radio_cheb.setText(_translate("Form", "Чеб. 2 роду"))
        # self.radio_sh_cheb_2.setText(_translate("Form", "Лагера"))
        self.custom_check.setText(_translate("Form", "Мультиплікативна форма"))
        self.groupBox_2.setTitle(_translate("Form", "Розмірність"))
        self.label_4.setText(_translate("Form", "X3"))
        self.label_2.setText(_translate("Form", "X1"))
        self.label_3.setText(_translate("Form", "X2"))
        self.label_5.setText(_translate("Form", "Y"))
        self.groupBox_4.setTitle(_translate("Form", "Степені"))
        self.label_6.setText(_translate("Form", "X1"))
        self.label_7.setText(_translate("Form", "X3"))
        self.label_8.setText(_translate("Form", "X2"))
        self.bruteforce_btn.setText(_translate("Form", "Обчислити"))
        self.bruteforce_btn.setStyleSheet("font-weight: bold; font-size: 14")
        #
        self.groupBox_7.setTitle(_translate("Form", "Додатково"))
        self.groupBox_7.setStyleSheet("font-weight: bold; font-size: 14")
        self.label_9.setText(_translate("Form", "Ваги:"))
        self.weights_box.setItemText(0, _translate("Form", "Average"))
        self.weights_box.setItemText(1, _translate("Form", "Scaled"))
        self.lambda_check.setText(_translate("Form", "3-блочні вирази"))
        #
        self.groupBox.setTitle(_translate("Form", "Дані"))
        self.groupBox.setStyleSheet("font-weight: bold; font-size: 14")
        self.line_output.setPlaceholderText(_translate("Form", "Вихідний файл"))
        self.select_output.setText(_translate("Form", "..."))
        self.line_input.setPlaceholderText(_translate("Form", "Вхідний файл"))
        self.select_input.setText(_translate("Form", "..."))
        item = self.tablewidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "Час"))
        # item.setBackground(QtGui.QColor(230, 249, 255))
        item = self.tablewidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Бортова напруга"))
        item = self.tablewidget.horizontalHeaderItem(2)
        item.setText(_translate("Form", "Запас\n" "палива"))
        item = self.tablewidget.horizontalHeaderItem(3)
        item.setText(_translate("Form", "Енергія\n" "АБ"))
        item = self.tablewidget.horizontalHeaderItem(4)
        item.setText(_translate("Form", "Стан\n" "фунціонування"))
        item = self.tablewidget.horizontalHeaderItem(5)
        item.setText(_translate("Form", "Ризик\n" "аварії"))
        item = self.tablewidget.horizontalHeaderItem(6)
        item.setText(_translate("Form", "Причина\n" "нештатної\n" "ситуації"))
        item = self.tablewidget.horizontalHeaderItem(7)
        item.setText(_translate("Form", "Рівень\n" "небезпеки"))
        __sortingEnabled = self.tablewidget.isSortingEnabled()
        self.tablewidget.setSortingEnabled(True)
        self.tablewidget.setSortingEnabled(__sortingEnabled)
        self.Current.setTitle(_translate("Form", "Поточний"))
        self.Current.setStyleSheet("font-weight: bold; font-size: 14")
        self.label_11.setText(_translate("Form", "Час"))
        self.label_12.setText(_translate("Form", "Бортова напруга"))
        self.label_13.setText(_translate("Form", "Запас палива"))
        self.label_15.setText(_translate("Form", "РДР"))
        self.label_14.setText(_translate("Form", "Енергія АБ"))
        self.groupBox_6.setTitle(_translate("Form", "Обробка"))
        self.groupBox_6.setStyleSheet("font-weight: bold; font-size: 14")
        self.label_10.setText(_translate("Form", "Кроки передбачення:"))
        self.exec_button.setText(_translate("Form", "Обчислити"))
        self.label.setText(_translate("Form", "Розмір вибірки (N_02)"))
