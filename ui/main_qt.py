

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout

class OpenRoboVisionApp(QWidget):

    """
    This class defines the GUI application based on qt for open-robo-vision project. Through this 
    GUI, user can interact with vision models specially adaptive models. Though user minimal 
    interactions, the AI should learn the task or adjust its behaviour.
    """

    def __init__(self):
        """
        Create an instance of OpenRoboVisionApp, create and structure the GUI element, and initialize 
        the variables.
        """

        super().__init__()
        self.setWindowTitle("Open Robo Vision")
        self.setGeometry(100, 100, 800, 800)

        # The main layout dividing the UI to settings bar at right and main views at left
        main_layout = QHBoxLayout(self)

        # Left section widget
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Top section in the left area
        left_top = QLabel("Left Top Section")
        left_layout.addWidget(left_top)
        
        # Bottom section in the left area
        left_bottom = QLabel("Left Bottom Section")
        left_layout.addWidget(left_bottom)
        
        # Right section widget (for demonstration, using a simple label)
        right_widget = QLabel("Right Section")
        
        # Add both left and right widgets to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpenRoboVisionApp()
    window.show()
    sys.exit(app.exec_())