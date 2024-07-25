
import math
import pickle
# Function to check if the difference between dates is more than 2 hours
from datetime import datetime, timedelta
import pickle
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx
from tqdm import tqdm
from graph_consts import Consts
from joblib import Parallel, delayed

def discretize_date(date):
    if date.hour<=3 or date.hour >=19:
        return "night"
    elif date.hour>=4 and date.hour<=11:
        return "morning" 
    return "Mid-Day"
def diff_greater_than_n_hours(time1, time2, hours):
    return (time2 - time1) > timedelta(hours=hours)

def diff_greater_than_n_sec(time1, time2, seconds=3):
    return (time2 - time1) > timedelta(seconds=seconds)

def create_voyages(df, camera_to_site):
    # Initialize a dictionary to hold the voyages
    voyages_dict = {}
    undefiend_ishitId={}
    # Iterate over the DataFrame, grouped by 'LpId'
    for lpid, group in  tqdm(df.groupby('LpId'), desc="Processing LpIds"):
        voyages = {}
        current_voyage = []
        last_date = None
        for index, row in group.iterrows():
            combined_datetime = row['CombinedDateTime']
            if last_date is not None:
                timestamp = pd.Timestamp(last_date)
                formatted_date = timestamp.strftime('%d/%m')
                if diff_greater_than_n_hours(last_date, combined_datetime, Consts['max_time_between_samples']):
                    if len(current_voyage) >= Consts['min_len_voyage']:
                        if formatted_date in voyages:
                            voyages[formatted_date].append(current_voyage)
                        else:
                            voyages[formatted_date] = [current_voyage]
                    current_voyage = []
            if last_date is None or diff_greater_than_n_sec(last_date, combined_datetime, Consts['min_time_between_samples']):
                date_time = discretize_date(combined_datetime)
                if row['IshitId'] in camera_to_site:
                    node = [
                        camera_to_site[row['IshitId']],
                        date_time,
                        combined_datetime,
                        row['x'],
                        row['y']
                    ]
                else:
                    undefiend_ishitId[row['IshitId']] = 1
                    continue
                if current_voyage!= []:
                    if  current_voyage[-1][0] == node[0]:
                        continue
                current_voyage.append(node)

            last_date = combined_datetime
        
        if current_voyage and len(current_voyage) >= Consts['min_len_voyage']:
            formatted_date = pd.Timestamp(last_date).strftime('%d/%m')
            if formatted_date in voyages:
                    voyages[formatted_date].append(current_voyage)
            else:
                    voyages[formatted_date] = [current_voyage]
        
        if voyages:
            voyages_dict[lpid] = voyages
    
    return voyages_dict,undefiend_ishitId
def parallel_create_voyages(df, camera_to_site, Consts):
    voyages_dict = {}
    undefined_ishitId = {}
    
    results = Parallel(n_jobs=-1)(
        delayed(process_group)(lpid, group, camera_to_site, Consts)
        for lpid, group in tqdm(df.groupby('LpId'), desc="Processing LpIds")
    )

    for lpid, voyages, undefined_ids in results:
        if voyages:
            voyages_dict[lpid] = voyages
        undefined_ishitId.update(undefined_ids)
    
    return voyages_dict, undefined_ishitId



def process_group(lpid, group, camera_to_site, Consts):
    voyages = {}
    current_voyage = []
    last_date = None
    undefined_ishitId = {}

    for index, row in group.iterrows():
        combined_datetime = row['CombinedDateTime']
        if last_date is not None:
            timestamp = pd.Timestamp(last_date)
            formatted_date = timestamp.strftime('%d/%m')
            if diff_greater_than_n_hours(last_date, combined_datetime, Consts['max_time_between_samples']):
                if len(current_voyage) >= Consts['min_len_voyage']:
                    if formatted_date in voyages:
                        voyages[formatted_date].append(current_voyage)
                    else:
                        voyages[formatted_date] = [current_voyage]
                current_voyage = []
        if last_date is None or diff_greater_than_n_sec(last_date, combined_datetime, Consts['min_time_between_samples']):
            date_time = discretize_date(combined_datetime)
            if row['IshitId'] in camera_to_site:
                node = [
                    camera_to_site[row['IshitId']],
                    date_time,
                    combined_datetime,
                    row['x'],
                    row['y']
                ]
            else:
                undefined_ishitId[row['IshitId']] = 1
                continue
            if current_voyage != []:
                if current_voyage[-1][0] == node[0]:
                    continue
            current_voyage.append(node)

        last_date = combined_datetime
    
    if current_voyage and len(current_voyage) >= Consts['min_len_voyage']:
        formatted_date = pd.Timestamp(last_date).strftime('%d/%m')
        if formatted_date in voyages:
            voyages[formatted_date].append(current_voyage)
        else:
            voyages[formatted_date] = [current_voyage]
    
    return lpid, voyages, undefined_ishitId


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




def preprocess_dataframe(df):
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Convert 'Hour' column to datetime, specifying the format, then to timedelta
    df['Hour'] = pd.to_datetime(df['Hour'], format='%I:%M:%S %p').dt.strftime('%H:%M:%S')
    df['Hour'] = pd.to_timedelta(df['Hour'])

    # Combine 'Date' and 'Hour'
    df['CombinedDateTime'] = df['Date'] + df['Hour']
    df = df.sort_values(by=['LpId', 'CombinedDateTime']).copy()
    
    return df
def split_data(df,date):  
    # Convert 'Date' column to datetime
    # df['Date'] = pd.to_datetime(df['Date'])
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