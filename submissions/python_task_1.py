import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta

# Load the data
df = pd.read_csv(r"C:\Users\Rentorzo\Downloads\Satya\MapUp-Data-Assessment-F\datasets\dataset-1.csv")
df1 = pd.read_csv(r"C:\Users\Rentorzo\Downloads\Satya\MapUp-Data-Assessment-F\datasets\dataset-2.csv")

def generate_car_matrix(df):
    # Check the case of the column names
    columns_to_keep = ['id_1', 'id_2', 'car']
    missing_columns = [col for col in columns_to_keep if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Extract relevant columns
    car_data = df[columns_to_keep].copy()

    # Pivot the DataFrame to get the desired matrix structure
    car_matrix = car_data.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    for col in car_matrix.columns:
        car_matrix.at[col, col] = 0

    return car_matrix

# Generate the new DataFrame
car_matrix = generate_car_matrix(df)
print(car_matrix)


def get_type_count(df):
    # Create a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each car_type category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_count_sorted = dict(sorted(type_count.items()))

    return type_count_sorted

result = get_type_count(df)
print(result)

def get_bus_indexes(df: pd.DataFrame) -> list:
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indices = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indices.sort()

    return bus_indices

list = get_bus_indexes(df)
print(list)

def filter_routes(df) -> list:
    # Calculate the average of 'truck' values for each 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

filtered_routes = filter_routes(df)
print(filtered_routes)

def multiply_matrix(input_df):
    modified_df = input_df.copy()

    # Apply the specified logic to each value in the DataFrame
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

modified_result = multiply_matrix(new_df)
print(modified_result)


def check_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' to create a 'start_timestamp' column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' to create an 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Define a list of all days of the week
    all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a date range for a full 24-hour period for each day of the week
    full_24_hours = pd.date_range('00:00:00', '23:59:59', freq='1S')
    full_7_days = [day + ' ' + str(time) for day in all_days for time in full_24_hours]

    # Combine start and end timestamps for each row
    pair_timestamps = pd.concat([df['start_timestamp'], df['end_timestamp']])

    # Check if the pair timestamps cover a full 24-hour period and span all 7 days
    completeness_check = pair_timestamps.between(min(full_7_days), max(full_7_days))

    # Create a boolean series with a multi-index (id, id_2)
    result_series = completeness_check.groupby([df['id'], df['id_2']]).all()

    return result_series
result_series = check_timestamp_completeness(df1)
print(result_series)