
import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget,
                             QDialog, QFormLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QMessageBox, QSplitter,
                             QScrollArea)
import pandas as pd

from simulated_data import create_simulated_data


class AnnotationDialog(QDialog):
    def __init__(self, annotation_config, clear_callback, deactivate_callback, existing_data=None):
        super().__init__()
        self.setWindowTitle("Annotation Input")
        self.setGeometry(300, 300, 400, 200)
        self.annotation_config = annotation_config
        self.clear_callback = clear_callback
        self.deactivate_callback = deactivate_callback
        self.form_layout = QFormLayout()
        self.combos = {}
        self.create_dynamic_fields()
        if existing_data:
            self.load_existing_data(existing_data)
        submit_layout = QHBoxLayout()
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.accept)
        submit_layout.addWidget(self.submit_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)
        submit_layout.addWidget(cancel_button)
        self.form_layout.addRow(submit_layout)
        self.setLayout(self.form_layout)

    def create_dynamic_fields(self):
        for field_name, field_config in self.annotation_config['labels'].items():
            label_text = field_config['description']
            choices = field_config['choices']
            combo = QComboBox()
            combo.addItems(map(str, choices))
            self.combos[field_name] = combo
            self.form_layout.addRow(QLabel(label_text), combo)

    def load_existing_data(self, existing_data):
        for field, value in existing_data.items():
            if field in self.combos:
                self.combos[field].setCurrentText(value)

    def get_annotation_data(self):
        return {field: combo.currentText() for field, combo in self.combos.items()}

    def closeEvent(self, a0, QCloseEvent=None):
        self.on_cancel()

    def on_cancel(self):
        self.clear_callback()
        self.deactivate_callback()
        self.reject()


class TimeseriesAnnotationTool(QMainWindow):

    def __init__(self, time_series_data, time_column, annotation_config):
        super().__init__()
        self.setWindowTitle("Synchronized Interval Selection with Dynamic Annotations")
        self.setGeometry(100, 100, 800, 600)
        self.time_column = time_column
        if isinstance(time_series_data, pd.DataFrame):
            self.time_series_df = time_series_data.copy().sort_index()
            self.x_data = self.time_series_df[self.time_column]
            del self.time_series_df[self.time_column]
        else:
            raise ValueError("Unsupported data type. Provide a DataFrame or a list of (x, y) tuples.")
        self.annotation_config = annotation_config
        self.axes = []
        self.figures = []
        splitter = QSplitter()
        left_side = QWidget(self)
        left_layout = QVBoxLayout(left_side)
        splitter.addWidget(left_side)
        self.index_list = QListWidget()
        self.unique_index = [tuple(map(lambda x: int(x) if isinstance(x, np.int64) else x, _)) for _ in self.time_series_df.index.unique()]
        self.index_list.addItems([",".join([str(_) for _ in i]) for i in self.unique_index])
        self.index_list.setCurrentItem(self.index_list.item(0))
        self.index_list.currentItemChanged.connect(self.on_index_selection_changed)
        left_layout.addWidget(QLabel("Select Index: "+(",".join(time_series_data.index.names))))
        left_layout.addWidget(self.index_list)
        right_side = QWidget(self)
        self.plot_layout = QVBoxLayout(right_side)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.plot_layout.addWidget(self.scroll_area)
        splitter.addWidget(right_side)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)
        self.selected_index = self.unique_index[0]
        self.plot_data(self.selected_index)
        self.load_annotations()

    def on_index_selection_changed(self):
        selected_row = self.index_list.currentRow()
        self.selected_index = self.unique_index[selected_row]
        self.plot_data(self.selected_index)
        self.plot_annotations()

    def plot_data(self, index=None):
        for fig_canvas in self.figures:
            self.scroll_layout.removeWidget(fig_canvas)
            fig_canvas.deleteLater()
        self.figures.clear()
        self.axes.clear()

        if index is not None:
            selected_row = self.time_series_df.loc[index]
            for column in self.time_series_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                canvas = FigureCanvas(fig)
                self.figures.append(canvas)
                self.axes.append(ax)
                ax.plot(self.x_data.loc[index], selected_row[column], label=f'{column}', color=np.random.rand(3, ))
                ax.set_title(f'{column}')
                ax.set_xlabel('Time (Index)')
                ax.set_ylabel(column)
                ax.legend()
                ax.grid(True)
                self.scroll_layout.addWidget(canvas)
        self.scroll_content.setLayout(self.scroll_layout)

    def get_annotation_color(self, annotation_data):
        for key in annotation_data:
            if key in self.annotation_config['labels']:
                return self.annotation_config['labels'][key]['color']
        return 'gray'

    def save_annotations(self):
        annotation_file = self.annotation_config['file']
        with open(annotation_file, 'w') as fp:
            json.dump(self.annotations, fp)

    def load_annotations(self):
        annotation_file = self.annotation_config['file']
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as fp:
                self.annotations = json.load(fp)
            for annotation_data in self.annotations:
                x_min = annotation_data['x_min']
                x_max = annotation_data['x_max']
                annotation_data['index'] = tuple(annotation_data['index'])
                color = self.get_annotation_color(annotation_data)
                for ax in self.axes:
                    rect = ax.axvspan(x_min, x_max, color=color, alpha=0.3)
                    self.rects.append(rect)
            self.plot_annotations()

    def plot_annotations(self):
        for rect in self.rects:
            rect.remove()
        self.rects.clear()
        for annotation in self.annotations:
            if annotation['index'] == self.selected_index:
                color = self.get_annotation_color(annotation)
                for ax in self.axes:
                    rect = ax.axvspan(annotation['x_min'], annotation['x_max'], color=color, alpha=0.3)
                    self.rects.append(rect)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    time_series_data, time_column = create_simulated_data()
    annotation_config = {
        'file': "annotations.json",
        'labels': {
            'hurry': {'description': 'Indicates urgency.', 'choices': ['No', 'Yes'], 'color': 'red'},
            'frustration': {'description': 'Indicates frustration.', 'choices': ['No', 'Yes'], 'color': 'orange'},
            'surprise': {'description': 'Indicates surprise.', 'choices': ['No', 'Yes'], 'color': 'blue'},
            'risk_evaluation': {'description': 'Risk level from 0 (low) to 3 (high)', 'choices': [0, 1, 2, 3], 'color': 'purple'}
        }
    }
    main_win = TimeseriesAnnotationTool(time_series_data, time_column, annotation_config)
    main_win.show()
    sys.exit(app.exec_())
