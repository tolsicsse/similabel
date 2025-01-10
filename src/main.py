import os.path
import sys
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QHBoxLayout, QListWidget, QListWidgetItem,
                             QScrollArea, QGridLayout, QSizePolicy, QSplitter)
from TimeseriesAnnotationTool import TimeseriesAnnotationTool
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QPen
import numpy as np
import pandas as pd

from simulated_data import create_simulated_data
from src.nnclr import fit_nnclr_model, get_similarities, prepare_model_to_load


class QSelectableLabel(QLabel):
    def __init__(self, parent=None):
        super(QSelectableLabel, self).__init__(parent)
        self.selected = False
        self.pixmap_selected: QPixmap = None
        self.pixmap_original: QPixmap = None
        self.line_width = 8

    def setPixmap(self, a0: QPixmap) -> None:
        super(QSelectableLabel, self).setPixmap(a0)
        if self.pixmap_selected is None:
            self.pixmap_original = a0
            rect = a0.rect()
            self.pixmap_selected = a0.copy()
            painter = QPainter(self.pixmap_selected)
            delta = int(self.line_width / 2)
            pen = QPen(Qt.blue, self.line_width)
            painter.setPen(pen)
            painter.drawRect(rect.adjusted(delta, delta, -delta, -delta))
            painter.end()

    def mousePressEvent(self, ev):
        if ev.modifiers() & Qt.ShiftModifier:
            if self.selected:
                self.setPixmap(self.pixmap_original)
                self.selected = False
            else:
                self.setPixmap(self.pixmap_selected)
                self.selected = True
        super(QSelectableLabel, self).mousePressEvent(ev)

class ThumbnailWidget(QLabel):
    """A widget that displays a thumbnail for a given annotation."""
    def __init__(self, annotation, x_data, y_data, color, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 100)

        self.annotation = annotation
        self.x_data = x_data
        self.y_data = y_data
        self.color = color

        # Create pixmap for drawing
        self.thumbnail_pixmap = QPixmap(self.size())
        self.plot_thumbnail()

    def plot_thumbnail(self):
        """Plot the thumbnail of the annotation."""
        self.thumbnail_pixmap.fill(Qt.white)  # Fill the pixmap with white background

        painter = QPainter(self.thumbnail_pixmap)
        pen = QPen(self.color, 1)  # Use the assigned color for the pen
        painter.setPen(pen)

        x_min = self.annotation['x_min']
        x_max = self.annotation['x_max']

        # Define plotting area
        width = self.thumbnail_pixmap.width()
        height = self.thumbnail_pixmap.height()

        # Mask the data within the x_min and x_max range
        mask = (self.x_data >= x_min) & (self.x_data <= x_max)
        x_plot = self.x_data[mask].astype(float)
        y_plot = self.y_data[mask].astype(float)

        # Normalize the data to fit the thumbnail size
        if len(x_plot) > 0:
            x_len = (x_max - x_min)
            x_len = x_len if x_len > 0 else 1
            x_normalized = (x_plot - x_min) / x_len * width
            y_len = (np.max(y_plot) - np.min(y_plot))
            y_len = y_len if y_len > 0 else 1
            y_normalized = (y_plot - np.min(y_plot)) / y_len  * height

            # Create a QPainterPath to draw the line
            path = QPainterPath()
            path.moveTo(int(x_normalized[0]), height - int(y_normalized[0]))  # Start point of the path

            # Add lines between all points in the normalized plot
            for j in range(1, len(x_normalized)):
                path.lineTo(int(x_normalized[j]), height - int(y_normalized[j]))  # Line to next point

            # Draw the path
            painter.drawPath(path)

        painter.end()
        self.setPixmap(self.thumbnail_pixmap)

    def mouseDoubleClickEvent(self, event):
        """Handle the double-click event."""
        # Trigger the double-click event and pass the annotation to the parent
        self.parent().parent().parent().parent().parent().parent().parent().on_thumbnail_double_clicked(self.annotation)

class AnnotationThumbnailViewer(QMainWindow):
    def __init__(self, time_series_df, time_column, annotation_config):
        super().__init__()
        self.setWindowTitle("Annotation Thumbnail Viewer")
        self.setGeometry(100, 100, 1000, 600)
        self.time_column = time_column
        self.annotation_config = annotation_config
        self.tool_win = TimeseriesAnnotationTool(time_series_df, self.time_column, self.annotation_config)  # Initialize the tool window variable
        self.annotations = self.tool_win.annotations
        self.time_series_df = self.tool_win.time_series_df
        self.x_data = self.tool_win.x_data
        self.model = None
        self.model_weights_file = self.annotation_config['file'].replace(".json", "") + ".weights.h5"
        self.sort_annotations()

        # Create a central widget with a horizontal layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create a splitter to allow resizing of the annotation list
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # List widget for selecting annotation types and timeseries
        self.annotation_list = QListWidget()
        splitter.addWidget(self.annotation_list)  # Add annotation list to the splitter

        # Create a widget to hold time series thumbnails and add it to the splitter
        annotation_images_widget = QWidget()
        layout_annotation_images = QVBoxLayout(annotation_images_widget)
        self.label_layout = QHBoxLayout()
        layout_annotation_images.addLayout(self.label_layout)  # Add the label layout above the scroll area

        # Scrollable area for displaying thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_area_widget)  # Use a grid layout for arranging thumbnails
        self.scroll_area.setWidget(self.scroll_area_widget)
        layout_annotation_images.addWidget(self.scroll_area)  # Add the scroll area below the labels

        splitter.addWidget(annotation_images_widget)  # Add the thumbnail area to the splitter
        splitter.setSizes([150, 850])  # Set initial sizes for the annotation list and thumbnails

        # Add items for each annotation type and choice
        self.populate_annotation_list()

        # Handle selection changes
        self.annotation_list.itemSelectionChanged.connect(self.on_annotation_selected)

        # Define colors for each time series
        self.series_colors = [
            Qt.red, Qt.green, Qt.blue, Qt.yellow,
            Qt.magenta, Qt.cyan, Qt.darkGray, Qt.darkGreen
        ]

    def populate_annotation_list(self):
        """Populate the list with annotation types and choices."""
        for annotation_type, config in self.annotation_config['labels'].items():
            for choice in config['choices']:
                item = QListWidgetItem(f"{annotation_type}: {choice}")
                item.setData(1, (annotation_type, choice))
                self.annotation_list.addItem(item)

    def on_annotation_selected(self):
        """Handle the event when an annotation type is selected."""
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            return

        selected_item = selected_items[0]
        annotation_type, choice = selected_item.data(1)

        # Filter annotations based on the selected type and choice (dummy implementation)
        filtered_annotations = [annotation for annotation in np.array(self.annotations)[self.sorted_annotation_indices]
                                if annotation.get(annotation_type) == choice]

        # Clear existing thumbnails
        for i in reversed(range(self.scroll_layout.count())):
            widget_to_remove = self.scroll_layout.itemAt(i).widget()
            self.scroll_layout.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()

        # Add labels for each time series column at the top (outside scroll area)
        for i in reversed(range(self.label_layout.count())):
            widget_to_remove = self.label_layout.itemAt(i).widget()
            self.label_layout.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()

        # Create and add labels for each DataFrame column
        for i, column in enumerate(self.time_series_df.columns):
            color = self.series_colors[i % len(self.series_colors)]
            label = QLabel(f'{column}')
            label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
            self.label_layout.addWidget(label)  # Add labels to the static layout above the scroll area

        # Add new thumbnails in a row for filtered annotations (dummy implementation)
        row = 0
        for annotation in filtered_annotations:
            col = 0  # Reset column for each annotation
            x_data = self.x_data.loc[annotation['index']].values
            if len(x_data) > 0:
                annotation_df = self.time_series_df.loc[annotation['index']]
                for i, column in enumerate(self.time_series_df.columns):
                    color = self.series_colors[i % len(self.series_colors)]
                    y_data = annotation_df[column].values
                    thumbnail = ThumbnailWidget(annotation, x_data, y_data, color, parent=self.scroll_area_widget)
                    self.scroll_layout.addWidget(thumbnail, row, col)  # Add thumbnails to a grid
                    col += 1
                row += 1  # Move to the next row for the next annotation

    def open_timeseries_annotation_tool(self, annotation):
        """Open the TimeseriesAnnotationTool with the selected annotation."""
        self.tool_win.selected_annotation = annotation
        self.tool_win.plot_annotations()

        # Set window modality to block the main app while the tool window is open
        self.tool_win.setWindowModality(Qt.ApplicationModal)
        self.tool_win.show()
        self.sort_annotations()
        self.on_annotation_selected()

    def on_thumbnail_double_clicked(self, annotation):
        """Handle the double-click event on a thumbnail."""
        self.open_timeseries_annotation_tool(annotation)

    def sort_annotations(self):
        time_series_data = self.time_series_df.copy()
        time_series_data[self.time_column] = self.x_data

        if self.model is None:
            if not os.path.exists(self.model_weights_file):
                self.model = fit_nnclr_model(time_series_data, self.time_column, self.annotations)
                self.model.save_weights(self.model_weights_file)
            else:
                self.model = prepare_model_to_load(time_series_data, self.time_column, self.annotations)
                self.model.load_weights(self.model_weights_file)

        similarities = get_similarities(self.model, time_series_data, self.time_column, self.annotations)

        def argsort_sim_mat(sm):
            idx = [np.argmax(np.sum(sm, axis=1))]
            for i in range(1, len(sm)):
                sm_i = sm[idx[-1]].copy()
                sm_i[idx] = -1
                idx.append(np.argmax(sm_i))  # b
            return np.array(idx)

        self.sorted_annotation_indices = argsort_sim_mat(similarities)

if __name__ == "__main__":
    # Create the application and main window
    app = QApplication(sys.argv)

    # Generate some sample time series data as a pandas DataFrame

    time_series_df, time_column = create_simulated_data()

    # Define the annotation configuration
    annotation_config = {
        'file': "annotations.json",
        'labels':
            {
                'hurry': {
                    'description': 'Indicates whether the driver feels a sense of urgency.',
                    'choices': ['No', 'Yes'],
                    'color': 'red'  # This is just an example, the color will be managed in get_annotation_color
                },
                'frustration': {
                    'description': 'Indicates whether the driver is experiencing frustration.',
                    'choices': ['No', 'Yes'],
                    'color': 'orange'
                },
                'surprise': {
                    'description': 'Indicates whether the driver is surprised by an event.',
                    'choices': ['No', 'Yes'],
                    'color': 'blue'
                },
                'risk_outcome': {
                    'description': 'Describes the outcome of a risky situation.',
                    'choices': ['No Risk', 'Near Miss', 'Accident'],
                    'color': 'purple'
                }
            }
    }

    viewer = AnnotationThumbnailViewer(time_series_df, time_column, annotation_config)
    viewer.show()

    sys.exit(app.exec_())