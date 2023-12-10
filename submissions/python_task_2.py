import pandas as pd
import networkx as nx
from datetime import datetime, time, timedelta
from scipy.spatial import distance_matrix
from geopy.distance import geodesic

df = pd.read_csv(r"C:\Users\Rentorzo\Downloads\Satya\MapUp-Data-Assessment-F\datasets\dataset-3.csv")

def calculate_distance_matrix(df, start_col='id_start', end_col='id_end', distance_col='distance'):
    # Create a dictionary to store distances between IDs
    distances = {}

    # Iterate through the DataFrame and populate the distances dictionary
    for index, row in df.iterrows():
        start_id = row[start_col]
        end_id = row[end_col]
        distance = row[distance_col]

        # Add distance from start to end
        distances[(start_id, end_id)] = distance

        # Add distance from end to start (bidirectional)
        distances[(end_id, start_id)] = distance

    # Create a list of unique IDs
    unique_ids = sorted(list(set(df[start_col].unique()) | set(df[end_col].unique())))

    # Initialize a 2D list for the distance matrix
    distance_matrix = [[0] * len(unique_ids) for _ in range(len(unique_ids))]

    # Populate the distance matrix with cumulative distances
    for i in range(len(unique_ids)):
        for j in range(len(unique_ids)):
            if i != j:
                # Find the distance between IDs if available, otherwise set to 0
                distance_matrix[i][j] = distances.get((unique_ids[i], unique_ids[j]), 0)

    # Convert the 2D list to a DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=unique_ids, columns=unique_ids)

    return distance_df

def calculate_distance_matrix(df):
    G = nx.DiGraph()

    # Populate the graph with edges and distances from the DataFrame
    for index, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
        # Add the reverse edge for bidirectional distances
        G.add_edge(row['id_end'], row['id_start'], distance=row['distance'])

    # Calculate cumulative distances using Floyd-Warshall algorithm
    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance')

    # Create a DataFrame from the distance matrix
    result_df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    # Set diagonal values to 0
    result_df.values[tuple([range(result_df.shape[0])] * 2)] = 0

    return result_df


# Example usage:
result_df = calculate_distance_matrix(df)

print(result_df)

def unroll_distance_matrix(distance_matrix):
    # Create an empty list to store unrolled data
    unrolled_data = []

    # Iterate over the rows and columns of the distance matrix
    for i in range(len(distance_matrix.index)):
        for j in range(len(distance_matrix.columns)):
            id_start = distance_matrix.index[i]
            id_end = distance_matrix.columns[j]
            distance = distance_matrix.iloc[i, j]

            # Append the data to the list
            unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

unrolled_df = unroll_distance_matrix(result_df)


def find_ids_within_ten_percentage_threshold(distance_df, reference_id):
    # Filter rows where id_start matches the reference value
    reference_rows = distance_df[distance_df['id_start'] == reference_id]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds for the threshold (10%)
    lower_bound = average_distance - 0.1 * average_distance
    upper_bound = average_distance + 0.1 * average_distance

    # Filter rows within the 10% threshold
    within_threshold = distance_df[
        (distance_df['id_start'] != distance_df['id_end']) &  # Exclude same id_start to id_end
        (distance_df['distance'] >= lower_bound) &
        (distance_df['distance'] <= upper_bound)
    ]

    # Get unique values from id_start column within the threshold
    unique_ids_within_threshold = sorted(within_threshold['id_start'].unique())

    return unique_ids_within_threshold

# Example usage:
# Assuming you have the unrolled_df from the previous question
reference_id = 1001408  # You can set the desired reference ID
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(ids_within_threshold)


def calculate_toll_rate(df)->pd.DataFrame():
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Add columns for each vehicle type with their respective rate coefficients
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient
    return df

result_df_with_toll_rates = calculate_toll_rate(unrolled_df)
print(result_df_with_toll_rates)

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                           (time(10, 0, 1), time(18, 0, 0)),
                           (time(18, 0, 1), time(23, 59, 59))]

    weekend_time_range = (time(0, 0, 0), time(23, 59, 59))

    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Function to calculate toll rates based on time
    def calculate_toll_rate(row):
        if row['start_time'].weekday() < 5:  # Weekdays
            for i, (start_time, end_time) in enumerate(weekday_time_ranges):
                if start_time <= row['start_time'].time() <= end_time:
                    return row['distance'] * row['rate_coefficient'] * weekday_discount_factors[i]
        else:  # Weekends
            return row['distance'] * row['rate_coefficient'] * weekend_discount_factor

    # Expand the DataFrame to cover all time intervals within a day
    time_intervals = pd.date_range('2023-01-01', periods=24, freq='H').time
    expanded_df = pd.DataFrame([(datetime.combine(datetime.min, start), datetime.combine(datetime.min, end)) for start in time_intervals for end in time_intervals],
                               columns=['start_time', 'end_time'])

    # Merge the original DataFrame with the expanded one
    merged_df = pd.merge(df, expanded_df, how='cross')

    # Calculate the base rates for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        merged_df[f'{vehicle_type}_rate_coefficient'] = rate_coefficient

    # Apply the toll rate calculation function
    merged_df['toll_rate'] = merged_df.apply(calculate_toll_rate, axis=1)

    # Extract day and time components
    merged_df['start_day'] = merged_df['start_time'].apply(lambda x: x.strftime('%A'))
    merged_df['end_day'] = merged_df['end_time'].apply(lambda x: x.strftime('%A'))

    # Drop unnecessary columns and rearrange columns
    result_df = merged_df[['start_day', 'start_time', 'end_day', 'end_time', 'id_start', 'id_end', 'toll_rate'] +
                          [f'{vehicle_type}_rate_coefficient' for vehicle_type in rate_coefficients.keys()]]

    return result_df

result_with_time_df = calculate_time_based_toll_rates(result_df_with_toll_rates)
print(result_with_time_df)