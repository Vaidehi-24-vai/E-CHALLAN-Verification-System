import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,  QMessageBox, QLineEdit, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QPainter
from PyQt5.QtCore import Qt, QTimer,QThread,  pyqtSignal
from detect import run
import cv2
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from text_recognition import perform_ocr  

class ParkingRulesWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PARKING RULES')
        self.setGeometry(100, 100, 600, 400)

        title_label = QLabel('PARKING RULES', self)
        title_label.setStyleSheet("color: Blue; font-size: 50px; font-weight: bold;")
        title_label.setGeometry(700, 10, 500, 40)

        rules_text = """
       BASIC PARKING RULES
       Do not park any vehicle on the part of a road where traffic is flowing.
       Do not park in such a place where one does not have a clear view for at least 50 meters in both the directions.
       Do not park in a place that blocks a vehicle already parked, a sidewalk, crosswalk, pedestrian crossing or road entrance.
       Do not park your vehicle on the bus bays or bus stops
       Do not park near the public entrance to a hotel, theatre or public hall when it is open to the public.
       Do not park near any intersection or a round-about.
       Do not park the vehicle in unauthorized parking areas.
       Before opening the door of the parked vehicle, look around. Close the door the moment one gets down.
       Park the vehicle on the left side in the direction of the traffic.

       PARKING AT NIGHT
       Try to park vehicles in a lighted place at night. If it is complete dark, turn the parking light on.
       Precautions While Passing Parked Vehicles	
       Be careful while passing by a parked vehicle. Never open the door without looking.
       Beware, a child might be playing and can come in front of your vehicle all of a sudden
       
       ROAD SIDE STOP
       In an emergency, if you have to stop by the side of the road, make sure the stop is very short.
  	   Give signal that you want to pull over and check your mirrors and blind spot to see when the way is clear.
  	   Steer to the side of the road, stopping very close to the curb or edge of the road.
  	   Put on the four way emergency lights.
  	   Apply handbrakes.
       
       WHAT NOT DO WHILE PARKING
       Do not park:
  	   On a foot path.
  	   Near traffic crossing, round-about or a turn.
  	   On the main road.
  	   Where your vehicle obstructs traffic.
  	   On wrong side of the road.
  	   Where parking is prohibited.  
        """
        rules_label = QLabel(rules_text, self)
        rules_label.setStyleSheet("color: Black; font-size: 20px; font-weight: bold;")
        rules_label.setGeometry(5, 50, 1800, 1000)
        rules_label.setWordWrap(True)

        img_label = QLabel(self)
        pmp = QPixmap("prules.jpg") 
        if pmp.isNull():
            QMessageBox.warning(self, "Error", "Failed to load the image.")
        d1_width = 700
        d1_height = 400
        scaled_pmap = pmp.scaled(d1_width, d1_height)
        img_label.setPixmap(scaled_pmap)
            #img_label.setPixmap(pmp)
        img_label.setGeometry(1200, 260, 700, 600) 

class VerificationThread(QThread):
    verification_completed = pyqtSignal(str, str)

    def __init__(self, image_path,number_plate):
        super().__init__()
        self.image_path = image_path
        self.number_plate = number_plate

    def run(self):
        result, detected_image_path = run(image_path=self.image_path, number_plate=self.number_plate)
        self.verification_completed.emit(result,  detected_image_path)

class VerificationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.detected_image_path = None
        self.setWindowTitle('Verification')
        self.setGeometry(5, 30, 2000, 1200)
        self.validity_label = QLabel(self)  # Initialize validity_label attribute
        self.detected_image_label = QLabel(self)
        self.initUI()

    def initUI(self):
        self.result_label = QLabel("VERIFICATION RESULT: ", self)
        self.result_label.setStyleSheet("font-size: 30px; color: green; font: bold; ")
        self.result_label.setGeometry(60, 590, 390, 50)

        number_plate_label = QLabel("Enter Number Plate Number:", self)
        number_plate_label.setStyleSheet("font-size: 20px; color: blue; font: bold; ")
        number_plate_label.setGeometry(20, 220, 500, 100)

        self.number_plate_field = QLineEdit(self)
        self.number_plate_field.setStyleSheet("font-size: 20px; color: black; font: bold; ")
        self.number_plate_field.setGeometry(320, 250, 360, 40)

        self.select_image_label = QLabel("Please Select an Image File (jpg, png, jpeg): ", self)
        self.select_image_label.setStyleSheet("font-size: 20px; color: blue; font: bold; ")
        self.select_image_label.setGeometry(20, 320, 500, 100)

        self.file_name_field = QLineEdit(self)
        self.file_name_field.setReadOnly(True)
        self.file_name_field.setStyleSheet("font-size: 20px; color: black; font: bold; ")
        self.file_name_field.setGeometry(480, 350, 360, 40)

        self.choose_button = QPushButton("CHOOSE FILE", self)
        self.choose_button.setStyleSheet("background-color: orange; color: black;")
        self.choose_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.choose_button.clicked.connect(self.choose_file)
        self.choose_button.setGeometry(220, 460, 150, 60)
 
        self.v_button = QPushButton("VERIFY", self)
        self.v_button.setStyleSheet("background-color: orange; color: black;")
        self.v_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.v_button.clicked.connect(self.v_image)
        self.v_button.setGeometry(460, 460, 150, 60)

        self.detected_image_label = QLabel(self)
        self.detected_image_label.setGeometry(1000, 200, 800, 800)
        
        s_image_label = QLabel(self)
        p_map = QPixmap("Untitled.jpeg")
        if p_map.isNull():
            QMessageBox.warning(self, "Error", "Failed to load echallan.jpeg")
        d1_width = 600
        d1_height = 200
        scaled_pmap = p_map.scaled(d1_width, d1_height)
        s_image_label.setPixmap(scaled_pmap)
        s_image_label.setGeometry(700, 0, 800, 200)
        #self.s_image_label = QLabel(self)
        #self.s_image_label.setGeometry(20, 100, 480, 400) 

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_name:
            self.file_name_field.setText(file_name)
               
    def v_image(self):
        if self.file_name_field.text() and self.number_plate_field.text():
            image_path = self.file_name_field.text()
            number_plate = self.number_plate_field.text()
            self.verification_thread = VerificationThread(image_path,number_plate)
            self.verification_thread.verification_completed.connect(self.display_verification_result)
            self.verification_thread.start()
            
        else:
            QMessageBox.warning(self, "Error", "Image And Number Plate Fields Cannot Be Empty.")
    
    def clear_previous_result(self):
        # Clear previous result label
        self.validity_label.clear()
        self.validity_label.hide()

    def display_verification_result(self, detected_image_path, result):
        self.clear_previous_result()
        if result is not None:
            if "Valid" in result:
               QMessageBox.information(self, "Verification Result","Vehicle with the given number plate is Valid Parking." )
               parking_validity = "Valid"
            elif "Invalid" in result:
                 QMessageBox.warning(self, "Verification Result", "Vehicle with the given number plate is Invalid Parking.")
                 parking_validity = "Invalid"
            else:
                QMessageBox.warning(self, "Verification Result", "Vehicle with the given number plate is not found.")
                parking_validity = "Not Found"

            self.validity_label = QLabel(f"{parking_validity} Parking.", self)
            self.validity_label.setStyleSheet("font-size: 20px; color: black; font: bold; ")
            self.validity_label.setGeometry(60, 660, 200, 50)
            self.validity_label.show()  
        else:
            QMessageBox.warning(self, "Error", "Verification failed.") 
 
        if detected_image_path:
           try:
               self.detected_image_path = detected_image_path 
               #print("Detected Image Path:", detected_image_path)
               pixmap = QPixmap(detected_image_path)
               if not pixmap.isNull():
                   scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
                   self.detected_image_label.setPixmap(scaled_pixmap)
                   self.detected_image_label.show()
               else:
                   QMessageBox.warning(self, "Error", "Failed to load detected image.")
           except Exception as e:
               QMessageBox.warning(self, "Error", f"Failed to load detected image: {str(e)}")
        else:
          QMessageBox.warning(self, "Error", "Detected image path is empty.")
class MarqueeLabel(QLabel):
    def __init__(self, text):
        super().__init__()
        self.setText(text)
        self.setStyleSheet("background-color: aqua; color: black; padding: 10px;")
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setFont(QFont("Arial", 16, QFont.Bold))
        self.direction = 1  
        self.timer = self.startTimer(8)
        self.offset = 0

    def timerEvent(self, event):
        if self.offset >= self.width() - self.fontMetrics().width(self.text()):
            self.direction = -1
        elif self.offset <= 0:
            self.direction = 1
        self.offset += self.direction
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.drawText(self.offset, self.height() // 2, self.text())

class Login1(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.verification_page = None
        self.parking_rules_window = None 

    def initUI(self):
        self.setWindowTitle('Login')
        self.setGeometry(5, 30, 2000, 1200)

        parking_rules_button = QPushButton('PARKING RULES', self)
        parking_rules_button.setStyleSheet("background-color: yellow; color: black; font-size: 17px; font-weight: bold; font-family: Arial;")
        parking_rules_button.setGeometry(130, 700, 220, 60)  
        parking_rules_button.clicked.connect(self.show_parking_rules)

        image_label = QLabel(self)
        pixmap = QPixmap("traffic-highway.jpeg")
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Failed to load traffic-highway.jpeg")
        desired_width = 1400
        desired_height = 1000
        pixmap = pixmap.scaled(desired_width, desired_height)
        image_label.setPixmap(pixmap)
        image_label.setGeometry(500, 100, 1500, 870)  

        img1 = QLabel(self)
        pix1 = QPixmap("challan.png")
        if pix1.isNull():
            QMessageBox.warning(self, "Error", "Failed to load challan.png")
        d_width = 500
        d_height = 500
        pix1 = pix1.scaled(d_width, d_height)
        img1.setPixmap(pix1)
        img1.setGeometry(10, 100, 485, 400)
       
        challan_button = QPushButton('CHALLAN VERIFICATION', self)
        challan_button.setStyleSheet("background-color: yellow; color: black; font-size: 17px; font-weight: bold; font-family: Arial;")
        challan_button.setGeometry(130, 600, 220, 60)  
        challan_button.clicked.connect(self.challan_verification_clicked)

        parking_rules_button = QPushButton('PARKING RULES', self)
        parking_rules_button.setStyleSheet("background-color: yellow; color: black; font-size: 17px; font-weight: bold; font-family: Arial;")
        parking_rules_button.setGeometry(130, 700, 220, 60)  # Set button position and size
        #challan_button.clicked.connect(self.challan_verification_clicked)
        parking_rules_button.clicked.connect(self.show_parking_rules) 

        #challan_button = QPushButton('HELP', self)
        #challan_button.setStyleSheet("background-color: yellow; color: black; font-size: 17px; font-weight: bold; font-family: Arial;")
        #challan_button.setGeometry(130, 800, 220, 60)  # Set button position and size

        
        marquee_label = MarqueeLabel("SMART TRAFFIC OVERSIGHT SYSTEM")
        marquee_label.setGeometry(10, 10, 580, 50) 
        marquee_label.setFont(QFont("Arial", 16, QFont.Bold)) 

        
        layout = QVBoxLayout()
        layout.addWidget(marquee_label)
        layout.addStretch(1)
        self.setLayout(layout)

    def challan_verification_clicked(self):
        if not self.verification_page:
            self.verification_page = VerificationPage()
        self.verification_page.show()

    def show_parking_rules(self):
        if not self.parking_rules_window:
            self.parking_rules_window = ParkingRulesWindow()
        self.parking_rules_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = Login1()
    login_window.show()
    #verification_page = VerificationPage()
    #verification_page.show()
    sys.exit(app.exec_())
