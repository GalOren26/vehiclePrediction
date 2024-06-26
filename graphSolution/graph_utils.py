
import math
import pickle
# Function to check if the difference between dates is more than 2 hours
from datetime import datetime, timedelta
import pickle
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx

from graph_consts import Consts


def discretize_date(date):
    if date.hour<=3 or date.hour >=19:
        return "night"
    elif date.hour>=4 and date.hour<=11:
        return "morning" 
    return "Mid-Day"
def create_voyages_by_name(df):
    def diff_greater_than_n_hours(time1, time2,hours=1):
        return (time2 - time1) > timedelta(hours=hours)


    def diff_greater_than_n_sec (time1, time2,seconds=3):
        return (time2 - time1) > timedelta(seconds=seconds)

        # Convert 'date' column to datetime objects
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by 'LpId' and 'date'
    df = df.sort_values(by=['LpId', 'Date']).copy()

    # Initialize a dictionary to hold the voyages
    voyages_dict = {}
    # Iterate over the DataFrame, grouped by 'LpId'
    for lpid, group in df.groupby('LpId'):
        voyages = {} # List to hold voyages for this 'LpId'
        current_voyage = []  # List to hold the current voyage's data points
        last_date = None  # Variable to keep track of the last date in the current voyage
        
        for index, row in group.iterrows():
            time_obj =  datetime.strptime(row['Hour'], '%I:%M:%S %p')
            combined_datetime = row['Date'].replace(hour=time_obj.hour, minute=time_obj.minute, second=time_obj.second)
            row['Date']=combined_datetime
    
            if last_date is not None:
                timestamp = pd.Timestamp(last_date)
                # Format to day/month
                formatted_date = timestamp.strftime('%d/%m')
                if diff_greater_than_n_hours(last_date, row['Date'],Consts['min_time_between_samples']):
                    if len(current_voyage) >= Consts['min_len_voyage']:
                        if formatted_date in voyages:
                            voyages[formatted_date].append(current_voyage)
                        else:
                            voyages[formatted_date]=[current_voyage]
                    current_voyage = []
            if last_date is None or diff_greater_than_n_sec(last_date, row['Date'], Consts['min_time_between_samples']):
                    date_time=discretize_date(row['Date'])
                    current_voyage.append([row['externalCameraId'],date_time,row['Date']])
            last_date = row['Date']
            # Add the last voyage if it's not empty
        if current_voyage and len(current_voyage) >= Consts['min_len_voyage']:
                if formatted_date in voyages:
                    voyages[formatted_date].append(current_voyage)
                else:
                    voyages[formatted_date]=[current_voyage]
        if voyages:
            voyages_dict[lpid] = voyages
    return voyages_dict


def save_dict_to_pickle(data_dict, filename):
    """
    Saves a dictionary to a JSON file.
    
    :param data_dict: Dictionary to be saved.
    :param filename: Filename for the JSON file.
    """
    try:
        with open(filename, 'wb') as pickle_file:
          pickle.dump(data_dict, pickle_file)
        print(f"Dictionary saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save dictionary: {str(e)}")
        

def load_dict_from_pickle(filename):
    """
    Loads a dictionary from a JSON file.
    
    :param filename: Filename of the JSON file to be read.
    :return: Loaded dictionary, or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as pickle_file:
           loaded_dict = pickle.load(pickle_file)
        print(f"Dictionary loaded successfully from {filename}")
        return loaded_dict
    except FileNotFoundError:
        print(f"No file found with the name {filename}. Please check the filename and try again.")
    except Exception as e:
        print(f"Failed to load dictionary: {str(e)}")
    return None
def update_edge_counts(edges_count, start_node, end_node):
    if start_node not in edges_count:
        edges_count[start_node] = {}
    if end_node not in edges_count[start_node]:
        edges_count[start_node][end_node] = 0
    edges_count[start_node][end_node] += 1

def build_graph(edges_count):
    G = nx.DiGraph()
    for start_node, destinations in edges_count.items():
        total_traversals = sum(destinations.values())
        
        for end_node, count in destinations.items():
            weight = count / total_traversals
            G.add_edge(start_node, end_node, weight=weight)
    return G
def split_data(df,date='1/12/2023'):  
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Define the cutoff date
    cutoff_date = pd.to_datetime(date, format='%d/%m/%Y')
    # Split the DataFrame into before and after the cutoff date
    before_cutoff = df[df['Date'] < cutoff_date]
    after_cutoff = df[df['Date'] >= cutoff_date]
    return before_cutoff, after_cutoff

def distance_between_nodes(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 

    # Radius of Earth in meters. Use 6371000 for meters or 6371 for kilometers
    r = 6371000 

    # Calculate the result
    return c * r