import sys

from PyQt5.QtWidgets import QApplication

from main import AnnotationThumbnailViewer
import pandas as pd

import pandas as pd
import json


def convert_label_data_to_annotations(label_data, time_column, label_cols):
    """
    Convert label data with index to the annotation JSON format.

    Args:
    - label_data: DataFrame with 'start_time' and label columns
    - time_column: The name of the time column (e.g., 'start_time')
    - label_cols: The list of label columns (e.g., ['hurry', 'frustration', 'surprise', 'risk_evaluation'])

    Returns:
    - annotations: List of dictionaries in the annotation format
    """
    annotations = []

    # Translate numeric labels to their string equivalents
    label_translation = {
        'hurry': {1: "Yes", 2: "No"},
        'frustration': {1: "Yes", 2: "No"},
        'surprise': {1: "Yes", 2: "No"},
        'risk_evaluation': {0: "No Risk", 1: "Low Risk", 2: "Moderate Risk", 3: "High Risk"}
    }

    # Group by subject, lap, and contiguous identical label values
    label_data['group'] = (label_data[label_cols] != label_data[label_cols].shift()).any(axis=1).cumsum()

    for (subject, lap, group), group_df in label_data.groupby([label_data.index.get_level_values(0),
                                                               label_data.index.get_level_values(1),
                                                               'group']):
        # Extract label values for the group
        hurry_value = group_df['hurry'].iloc[0]
        frustration_value = group_df['frustration'].iloc[0]
        surprise_value = group_df['surprise'].iloc[0]
        risk_value = group_df['risk_evaluation'].iloc[0]

        # Get the x_min (start_time min) and x_max (start_time max) for the group
        x_min = int(group_df[time_column].min())
        x_max = int(group_df[time_column].max())

        # Create the annotation dictionary
        annotation = {
            'hurry': label_translation['hurry'][hurry_value],
            'frustration': label_translation['frustration'][frustration_value],
            'surprise': label_translation['surprise'][surprise_value],
            'risk_evaluation': label_translation['risk_evaluation'][risk_value],
            'x_min': x_min,
            'x_max': x_max,
            'index': [subject, int(lap)]
        }

        annotations.append(annotation)

    return annotations

def read_cpmxai_data():
    # Example usage
    data = pd.read_excel("original_data/Feature_Track.xlsx").dropna() #TODO how to handle missing values
    data.set_index(['subject', 'lap'], inplace=True)
    time_column = 'start_time'
    label_cols = ['hurry', 'frustration', 'surprise', 'risk_evaluation']
    data_cols = [
    "max_speed",
    "avg_speed",
    "std_speed",
    "yaw",
    "yaw_rate",
    "roll",
    "roll_rate",
    "pitch",
    "pitch_rate",
    "lat_acce",
    "long_acce",
    "vert_acce",
    "avg_acce_pedal_pos",
    "std_acce_pedal_pos",
    "avg_steer_angle",
    "std_steer_angle",
    "hr",
    "hrv_lf",
    "hrv_hf",
    "hrv_lfhf_ratio",
    "gsr_tonic",
    "gsr_phasic",
    "gsr_peaks",
    "EBRmean",
    "BDmean",
    "ThetaFrontal",
    "ThetaParietal",
    "AlphaFrontal",
    "AlphaParietal",
    "LowerAlphaFrontal",
    "LowerAlphaParietal",
    "UpperAlphaFrontal",
    "UpperAlphaParietal",
    "BetaFrontal",
    "BetaParietal",
    "LowerBetaFrontal",
    "LowerBetaParietal",
    "UpperBetaFrontal",
    "UpperBetaParietal"
    ]

    # Selecting only the required columns
    label_data = data[[time_column] + label_cols]

    # Convert the label data to annotation format
    annotations = convert_label_data_to_annotations(label_data, time_column, label_cols)

    # Optionally, save the annotations to a JSON file
    with open('cpmxai_annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)

    return data[[time_column] + data_cols], time_column, 'cpmxai_annotations.json'

if __name__ == "__main__":
    # Create the application and main window
    app = QApplication(sys.argv)

    # Generate some sample time series data as a pandas DataFrame

    time_series_df, time_column, annotation_file = read_cpmxai_data()

    # Define the annotation configuration
    annotation_config = {
        'file': annotation_file,
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
                    'choices': ["No Risk", "Low Risk", "Moderate Risk", "High Risk"],
                    'color': 'purple'
                }
            }
    }


    viewer = AnnotationThumbnailViewer(time_series_df, time_column, annotation_config)
    viewer.show()

    sys.exit(app.exec_())