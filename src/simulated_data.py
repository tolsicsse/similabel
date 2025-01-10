import pandas as pd
import numpy as np

def create_simulated_data():
    # Define number of drivers and laps
    num_drivers = 10
    num_laps = 7

    # Create lists to hold data
    driver_ids = []
    lap_numbers = []
    start_times = []

    # Generate time series data for each driver
    for driver_id in range(1, num_drivers+1):
        for lap in range(1, num_laps+1):
            current_time = np.random.uniform(0, 10)
            for _ in range(np.random.randint(90, 120, 1)[0]):
                driver_ids.append(driver_id)
                lap_numbers.append(lap)
                # Random start time for each lap, starting close to zero and increasing
                start_times.append(current_time)
                current_time += np.random.uniform(60, 90)  # Lap times between 60-90 seconds

    # Create DataFrame with driver_id and lap_number as the index
    df = pd.DataFrame({
        'driver_id': driver_ids,
        'lap_number': lap_numbers,
        'start_time_sec': start_times
    })

    # Set driver_id and lap_number as a multi-index
    df.set_index(['driver_id', 'lap_number'], inplace=True)

    # Generate random ECG data for each lap of each driver
    np.random.seed(42)  # For reproducible random values
    ecg_avg = np.random.uniform(60, 120, size=len(df))  # Average heart rate between 60-120 bpm
    ecg_min = ecg_avg - np.random.uniform(5, 20, size=len(df))  # Min heart rate slightly lower than average
    ecg_max = ecg_avg + np.random.uniform(5, 20, size=len(df))  # Max heart rate slightly higher than average

    # Add ECG columns to the DataFrame
    df['ecg_avg'] = ecg_avg
    df['ecg_min'] = ecg_min
    df['ecg_max'] = ecg_max

    # Adding car-related measurements: speed, acceleration, reaction time, and driver focus level
    car_speed_kmh = np.random.uniform(150, 200, size=len(df))  # Speed in km/h
    car_acceleration_ms2 = np.random.uniform(2, 4, size=len(df))  # Acceleration in m/s²
    driver_reaction_time_sec = np.random.uniform(0.2, 1.0, size=len(df))  # Reaction time in seconds
    driver_focus_level = np.random.uniform(50, 100, size=len(df))  # Focus level in percentage

    # Add car-related columns to the DataFrame
    df['car_speed_kmh'] = car_speed_kmh
    df['car_acceleration_ms2'] = car_acceleration_ms2
    df['driver_reaction_time_sec'] = driver_reaction_time_sec
    df['driver_focus_level'] = driver_focus_level

    # Display the first 15 rows of the updated DataFrame
    print(df.head(15))

    return df, 'start_time_sec'
