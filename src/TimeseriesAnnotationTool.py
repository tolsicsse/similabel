import json
import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import SpanSelector
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget,
                             QDialog, QFormLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QMessageBox, QSplitter,
                             QScrollArea, QSizePolicy)
import pandas as pd

from simulated_data import create_simulated_data


class AnnotationDialog(QDialog):
    """Custom dialog to input dynamic annotation details."""
    def __init__(self, annotation_config, clear_callback, deactivate_callback, existing_data=None):
        super().__init__()
        self.setWindowTitle("Annotation Input")
        self.setGeometry(300, 300, 400, 200)

        # Store the annotation configuration
        self.annotation_config = annotation_config
        self.clear_callback = clear_callback  # Callback to clear rectangles if canceled
        self.deactivate_callback = deactivate_callback  # Callback to deactivate SpanSelector

        # Create a layout for the dialog
        self.form_layout = QFormLayout()

        # Dictionary to store the comboboxes for each label
        self.combos = {}

        # Dynamically create labels and choices from the annotation_config
        self.create_dynamic_fields()

        # Load existing data if provided
        if existing_data:
            self.load_existing_data(existing_data)

        # Submit button
        submit_layout = QHBoxLayout()
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.accept)  # Close dialog on click
        submit_layout.addWidget(self.submit_button)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel)  # Clear rectangles and deactivate on cancel
        submit_layout.addWidget(cancel_button)

        # Add layout to the dialog
        self.form_layout.addRow(submit_layout)
        self.setLayout(self.form_layout)

    def create_dynamic_fields(self):
        """Create dynamic fields based on the annotation configuration."""
        for field_name, field_config in self.annotation_config['labels'].items():
            label_text = field_config['description']
            choices = field_config['choices']

            # Create a combo box for the field
            combo = QComboBox()
            combo.addItems(map(str,choices))
            self.combos[field_name] = combo  # Store the combo box

            # Add label and combo box to the layout
            self.form_layout.addRow(QLabel(label_text), combo)

    def load_existing_data(self, existing_data):
        """Load existing annotation data into the dialog."""
        for field, value in existing_data.items():
            if field in self.combos:
                self.combos[field].setCurrentText(value)

    def get_annotation_data(self):
        """Return the selected annotation data."""
        return {field: combo.currentText() for field, combo in self.combos.items()}

    def closeEvent(self, a0, QCloseEvent=None):
        self.on_cancel()

    def on_cancel(self):
        """Handle the cancel action by clearing rectangles and closing the dialog."""
        self.clear_callback()  # Call the clear function
        self.deactivate_callback()  # Call the deactivate function
        self.reject()  # Close the dialog without saving

class TimeseriesAnnotationTool(QMainWindow):

    def __init__(self, time_series_data, time_column, annotation_config):
        super().__init__()
        self.setWindowTitle("Synchronized Interval Selection with Dynamic Annotations")
        self.setGeometry(100, 100, 800, 600)
        self.time_column = time_column
        # Check if the input data is a DataFrame or list of tuples
        if isinstance(time_series_data, pd.DataFrame):
            self.time_series_df = time_series_data.copy().sort_index()
            self.x_data = self.time_series_df[self.time_column]
            del self.time_series_df[self.time_column]
        else:
            raise ValueError("Unsupported data type. Provide a DataFrame or a list of (x, y) tuples.")

        # Store the annotation configuration
        self.annotation_config = annotation_config

        # Main layout with a splitter
        splitter = QSplitter()

        # Left side: List of selectable indices
        left_side = QWidget(self)
        left_layout = QVBoxLayout(left_side)
        splitter.addWidget(left_side)

        # Create the index selection list
        self.index_list = QListWidget()
        self.unique_index = [tuple(map(lambda x: int(x) if isinstance(x, np.int64) else x, _)) for _ in self.time_series_df.index.unique()]
        self.index_list.addItems([",".join([str(_) for _ in i]) for i in self.unique_index])  # Convert indices to string
        self.index_list.setCurrentItem(self.index_list.item(0))
        self.index_list.currentItemChanged.connect(self.on_index_selection_changed)
        left_layout.addWidget(QLabel("Select Index: "+(",".join(time_series_data.index.names))))
        left_layout.addWidget(self.index_list)

        # Right side: Scrollable area for multiple plots
        right_side = QWidget(self)
        self.plot_layout = QVBoxLayout(right_side)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # Add scroll area to the plot layout
        self.scroll_area.setWidget(self.scroll_content)
        self.plot_layout.addWidget(self.scroll_area)
        splitter.addWidget(right_side)

        splitter.setStretchFactor(1, 3)  # Give more space to the plot area

        # Set splitter as the central widget
        self.setCentralWidget(splitter)

        self.rects = []  # Store annotation rectangles
        self.annotations = []
        self.current_selection_rects = []
        self.figures = []
        self.axes = []

        # var to detect single vs double clicks and for dragging
        self.double_click_detected = False  # Flag to track double clicks
        self.dragging = False
        self.selected_annotation = None
        self.drag_start_x = None
        self.selected_index = self.unique_index[0]

        # Initialize plot data with the first index
        self.plot_data(self.selected_index)

        # Access the Matplotlib canvas from each FigureCanvas object
        for canvas in self.figures:
            canvas.mpl_connect('button_press_event', self.on_click)
            canvas.mpl_connect('motion_notify_event', self.on_drag)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('button_press_event', self.on_click)

        # Create span selectors for all subplots
        self.span_selectors = []
        for ax in self.axes:
            ss = SpanSelector(ax, self.on_span_select, 'horizontal', interactive=True, useblit=True,
                              onmove_callback=self.update_span)
            self.span_selectors.append(ss)

        self.load_annotations()

    def on_index_selection_changed(self):
        """Update plot when the selected index changes."""
        selected_row = self.index_list.currentRow()  # Get selected row index
        self.selected_index = self.unique_index[selected_row]
        self.plot_data(self.selected_index)
        self.plot_annotations()
        for canvas in self.figures:
            canvas.draw_idle()

    def plot_data(self, index=None):
        """Plot the time series data for the selected index."""
        # Clear previous figures from the layout
        for fig_canvas in self.figures:
            self.scroll_layout.removeWidget(fig_canvas)
            fig_canvas.deleteLater()
        self.figures.clear()

        if index is not None:
            selected_row = self.time_series_df.loc[index]

            for column in self.time_series_df.columns:
                # Create a new figure and canvas for each column
                fig, ax = plt.subplots(figsize=(8, 4))
                canvas = FigureCanvas(fig)
                self.figures.append(canvas)

                # Plot data for the current column
                ax.plot(self.x_data.loc[index], selected_row[column], label=f'{column}', color=np.random.rand(3, ))
                ax.set_title(f'{column}')
                ax.set_xlabel('Time (Index)')
                ax.set_ylabel(column)
                ax.legend()
                ax.grid(True)



                # Add the canvas to the scrollable layout
                self.scroll_layout.addWidget(canvas)

        # Update the layout
        self.scroll_content.setLayout(self.scroll_layout)

    def set_y_limits(self):
        """Set y-limits to the maximum values across all datasets."""
        for ax, (_, y) in zip(self.axes, self.time_series_df.T):
            max_y = np.max(np.abs(y))
            ax.set_ylim(-max_y, max_y)  # Set the y-axis limit to maximum (symmetric)

    def on_span_select(self, x_min, x_max):
        """Callback for finalizing the span selection."""
        if not self.double_click_detected:  # Only if a double-click has not been detected
            # Open an annotation input dialog with the dynamic annotation configuration
            dialog = AnnotationDialog(self.annotation_config, self.clear_current_selection_rects, self.deactivate_span_selectors)
            if dialog.exec_() == QDialog.Accepted:
                annotation_data = dialog.get_annotation_data()
                annotation_data['x_min'] = x_min
                annotation_data['x_max'] = x_max
                annotation_data['index'] = tuple(map(lambda x: int(x) if isinstance(x, np.int64) else x, self.selected_index))
                self.annotations.append(annotation_data)

                # Display the selected area
                for ax in self.axes:
                    # Get color for the annotation based on the selected choice
                    color = self.get_annotation_color(annotation_data)
                    rect = ax.axvspan(x_min, x_max, color=color, alpha=0.3)
                    self.rects.append(rect)  # Store reference to the new rectangle

                # Update the canvas to show the selection and annotations
                for canvas in self.figures:
                    canvas.draw()

                # Display annotation information in a message box
                annotation_str = "\n".join([f"{key}: {value}" for key, value in annotation_data.items()])
                QMessageBox.information(self, "Annotation Saved", f"Selected interval:\nX: [{x_min:.2f}, {x_max:.2f}]\n\n{annotation_str}")

                # Save annotations to CSV file
                self.save_annotations()

    def update_span(self, x_min, x_max):
        """Real-time callback to update all subplots with the same selection and fill the selection area."""
        self.clear_current_selection_rects()  # Clear previous rectangles for current selection
        if self.dragging:
            self.deactivate_span_selectors()
            for canvas in self.figures:
                canvas.draw_idle()
            return

        # Draw the selection as a filled rectangle on all subplots
        for ax in self.axes:
            color = "yellow"  # Color for the current selection
            rect = ax.axvspan(x_min, x_max, color=color, alpha=0.3)
            self.current_selection_rects.append(rect)  # Store reference to the new rectangle

        # Update the canvas in real-time
        for canvas in self.figures:
            canvas.draw_idle()

    def clear_current_selection_rects(self):
        """Clear the previous dynamically drawn rectangles for the current selection."""
        for rect in self.current_selection_rects:
            rect.remove()  # Remove the rectangle from the axes

        # Clear the list after removing the spans
        self.current_selection_rects.clear()
        for canvas in self.figures:
            canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse clicks on the plot to select or move annotations."""
        if event.inaxes:  # Check if the click is within the axes
            x_click = event.xdata
            if event.dblclick:  # If it's a double-click
                # Disable the SpanSelector (since double-click overrides selection)
                self.double_click_detected = True
                for ss in self.span_selectors:
                    ss.set_active(False)

                # Check if the click is within an existing annotation
                for annotation in self.annotations:
                    if annotation['x_min'] <= x_click <= annotation['x_max']:
                        self.open_edit_annotation(annotation)
                        break  # Only show the first matching annotation

                # Reset the flag after handling the double-click
                self.double_click_detected = False
            else:
                # Single click - check if user clicked on an existing annotation to start dragging
                for annotation in self.annotations:
                    if annotation['x_min'] <= x_click <= annotation['x_max']:
                        self.dragging = True
                        self.selected_annotation = annotation
                        self.drag_start_x = x_click
                        self.original_annotation = {s:annotation[s] for s in annotation}

                        self.plot_annotations()
                        for canvas in self.figures:
                            canvas.draw_idle()
                        break

                self.double_click_detected = False
                for ss in self.span_selectors:
                    ss.set_active(True)  # Re-enable SpanSelector after single click

    def open_edit_annotation(self, annotation):
        self.selected_annotation = annotation
        self.plot_annotations()
        for canvas in self.figures:
            canvas.draw_idle()
        # If double-clicked within the bounds, show annotation details
        dialog = AnnotationDialog(self.annotation_config, self.clear_current_selection_rects,
                                  self.deactivate_span_selectors, annotation)
        if dialog.exec_() == QDialog.Accepted:
            annotation_data = dialog.get_annotation_data()
            for _ in annotation_data:
                annotation[_] = annotation_data[_]
            self.save_annotations()
        self.selected_annotation = None
        self.plot_annotations()
        for canvas in self.figures:
            canvas.draw_idle()

    def on_drag(self, event):
        """Handle dragging of annotations."""
        if self.dragging and event.inaxes and self.selected_annotation:
            x_current = event.xdata
            delta = x_current - self.drag_start_x
            self.selected_annotation['x_min'] += delta
            self.selected_annotation['x_max'] += delta
            self.drag_start_x = x_current

            self.plot_annotations()

            for canvas in self.figures:
                canvas.draw()

    def plot_annotations(self):
        # Clear and redraw all annotations
        for rect in self.rects:
            rect.remove()
        self.rects.clear()

        for annotation in self.annotations:
            if annotation['index'] == self.selected_index:
                color = self.get_annotation_color(annotation)
                for ax in self.axes:
                    rect = ax.axvspan(annotation['x_min'], annotation['x_max'], color=color, alpha=0.3)
                    self.rects.append(rect)

                    if annotation == self.selected_annotation:
                        self.rects += [ax.axvspan(annotation['x_min'], annotation['x_max'], facecolor='none', edgecolor='black',
                                   linewidth=2, linestyle='--')]

    def on_release(self, event):
        """Handle mouse release to stop dragging."""
        if self.dragging:
            if self.selected_annotation['x_min'] != self.original_annotation['x_min']:
                reply = QMessageBox.question(self, "", "Do you want move\nthe annotation?", QMessageBox.Yes | QMessageBox.No)
                print(reply)
                if reply == QMessageBox.Yes:
                    self.save_annotations()  # Save updated annotations after moving them
                else: # cancel
                    for _ in self.original_annotation:
                        self.selected_annotation[_] = self.original_annotation[_]
            self.clear_current_selection_rects()
            self.selected_annotation = None
            self.drag_start_x = None
            self.plot_annotations()
            for canvas in self.figures:
                canvas.draw_idle()
            self.dragging = False

    def deactivate_span_selectors(self):
        """Deactivate all SpanSelectors."""
        for ss in self.span_selectors:
            ss.set_active(False)
            ss.set_visible(False)

    def save_annotations(self):
        """Save the annotations to a json file."""
        annotation_file = self.annotation_config['file']
        with open(annotation_file, 'w') as fp:
            json.dump(self.annotations, fp)

    def load_annotations(self):
        """Load annotations from a json file (if exists)."""
        annotation_file = self.annotation_config['file']
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as fp:
                self.annotations = json.load(fp)
            for annotation_data in self.annotations:
                # Highlight the saved intervals on the plot
                x_min = annotation_data['x_min']
                x_max = annotation_data['x_max']

                annotation_data['index'] = tuple(annotation_data['index'])
                # Get color for the annotation based on the loaded choice
                color = self.get_annotation_color(annotation_data)
                for ax in self.axes:
                    rect = ax.axvspan(x_min, x_max, color=color, alpha=0.3)
                    self.rects.append(rect)

            # Update the canvas to show the loaded annotations
            self.plot_annotations()
            for canvas in self.figures:
                canvas.draw()

    def get_annotation_color(self, annotation_data):
        """Get the color for the annotation based on its properties."""
        for key in annotation_data:
            if key in self.annotation_config:
                return self.annotation_config[key]['color']
        return 'gray'  # Default color if not found


if __name__ == "__main__":
    # Example usage with arbitrary number of time series
    app = QApplication(sys.argv)

    # Generate some sample time series data (list of (x, y) tuples)
    time_series_data, time_column = create_simulated_data()

    # Define the annotation configuration with colors
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
                'risk_evaluation': {
                    'description': 'Describes the outcome of a risky situation from 0 (no risk) to 3 (very risky)',
                    'choices': [0, 1, 2, 3],
                    'color': 'purple'
                }
            }
    }

    # Create and run the main application window
    main_win = TimeseriesAnnotationTool(time_series_data, time_column, annotation_config)
    main_win.show()
    sys.exit(app.exec_())
