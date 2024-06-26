import pandas as pd
import matplotlib.pyplot as plt
import os
from graph_utils import build_graph, create_voyages_by_name, load_dict_from_pickle, save_dict_to_pickle, split_data, update_edge_counts
from graph_consts import Consts
# Set the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# voyages = load_dict_from_pickle(voyages_path)
# Load data
detections_path=os.path.join(Consts['trail_path'],Consts['detections_path_name']) 
df = pd.read_csv(detections_path)
df_AllLprCamers= pd.read_csv(Consts['camera_path'])
camera_name_dict = df_AllLprCamers.set_index('ExternalCameraId')['Name'].to_dict()
# double direction of english hebrew camera names 
name_to_cameraid = pd.Series(df_AllLprCamers.ExternalCameraId.values, index=df_AllLprCamers.Name).to_dict()
cameraid_to_name = pd.Series(df_AllLprCamers.Name.values, index=df_AllLprCamers.ExternalCameraId).to_dict()


train,test=split_data(df)
train_voyages=create_voyages_by_name(train)
test_voyages=create_voyages_by_name(test)
# # save and load data to bot crearte each time 
# voyages_path = 'inffered_voyages_graph.pkl'
# save_dict_to_pickle(voyages,voyages_path)
# voyages = load_dict_from_pickle(voyages_path)
# Initialize a dictionary to hold graphs for each vehicle
graphs_second_order = {"all": {}}
graphs_first_order={"all":{}}
edges_count_first = {"all":{}}
edges_count_second = {"all":{}}
time_stats_nodes={}
for lp in train_voyages.keys():
    previous_node = None
    edges_count_first[lp]={}
    edges_count_second[lp]={}
    time_stats_nodes[lp]={}
    for date in train_voyages[lp]:
        for voyage in train_voyages[lp][date]:
            for idx in range(len(voyage) - 1):
                if  voyage[idx][0] not in time_stats_nodes[lp]:
                     time_stats_nodes[lp][voyage[idx][0]]={"times":[]}
                if voyage[idx][1] in time_stats_nodes[lp][voyage[idx][0]]:
                    time_stats_nodes[lp][voyage[idx][0]][voyage[idx][1]]+=1
                else :
                    time_stats_nodes[lp][voyage[idx][0]][voyage[idx][1]]=1
                time_stats_nodes[lp][voyage[idx][0]]["times"].append(voyage[idx][2])
                start_node_first = voyage[idx][0]
                end_node_first = voyage[idx + 1][0]
                update_edge_counts(edges_count_first[lp], start_node_first, end_node_first)
                update_edge_counts(edges_count_first['all'],start_node_first,end_node_first)
                if previous_node is not None:
                    start_node_second = (previous_node, start_node_first)
                    end_node_second = end_node_first
                    update_edge_counts(edges_count_second[lp], start_node_second, end_node_second)
                    update_edge_counts(edges_count_second['all'], start_node_second, end_node_second)
                previous_node = start_node_first
            # Handle end nodes for both order transitions
        last_node = voyage[-1][0]
        virtual_end_node = 'End_Node'
        update_edge_counts(edges_count_first[lp], last_node, virtual_end_node)
        update_edge_counts(edges_count_first['all'], last_node, virtual_end_node)
        if previous_node:
            last_node_second = (previous_node, last_node)
            update_edge_counts(edges_count_second[lp], last_node_second, virtual_end_node)
            update_edge_counts(edges_count_second['all'], last_node_second, virtual_end_node)
    # Construct graphs
    graphs_first_order[lp] = build_graph(edges_count_first[lp])
    graphs_second_order[lp] = build_graph(edges_count_second[lp])
    graphs={"graphs_first_order": graphs_first_order,"graphs_second_order": graphs_second_order,'edges_count_first': edges_count_first,"edges_count_second": edges_count_second}
voyages_path = 'graph_stats.pkl'
save_dict_to_pickle(graphs,voyages_path)

#######infernce part################################ 

graphs=load_dict_from_pickle(voyages_path)
edges_count_first,edges_count_second=graphs['edges_count_first'],graphs['edges_count_second']
predicted_results = {}
# min_count = Consts["min_count"]
ratio_threshold = Consts["ratio_threshold"]
def predict_next_node(voyage, idx, edges_count_first, edges_count_second, ratio_threshold=0.5):
    # Try to predict using second-order transitions with fallback to the first order
    current_node=voyage[idx][0]
    for i in range(1, 4):  # Look back 1 to 3 nodes
        if idx >= i:  # Check there are enough previous nodes
            previous_node = voyage[idx - i][0]
            pair = (previous_node, current_node)
            if pair in edges_count_second:
                total_second_order = sum(edges_count_second[pair].values())
                potential_next_node  = max(edges_count_second[pair], key=edges_count_second[pair].get)
                count_potential_next_node = edges_count_second[pair][potential_next_node]
                # total_first_order = sum(edges_count_first[current_node].values()) if current_node in edges_count_first else 0

                # Check minimum count and ratio conditions
                # if total_second_order >= min_count and (total_first_order == 0 or (total_second_order / total_first_order >= ratio_threshold)):
                if  count_potential_next_node / total_second_order >= ratio_threshold:
                    next_node = max(edges_count_second[pair], key=edges_count_second[pair].get)
                    return next_node, f'second_order-{i} steps back'
    
    # Fallback to first-order transitions
    if current_node in edges_count_first:
        next_node = max(edges_count_first[current_node], key=edges_count_first[current_node].get)
        return next_node, 'first_order'
    
    return None, 'none'  # If no prediction can be made


precision={}
precision['all']={"precision":0,
                   'total_number_of_nodes':0
                   }
for lp in test_voyages.keys():
    total_number_of_nodes=0
    precision[lp]={"precision":0,
                   'total_number_of_nodes':0
                   }
    #Todo drop this condiation affter wh have data to all car 
    if lp in train_voyages:
        predicted_results[lp] = {}
        for date in test_voyages[lp]:
            predicted_results[lp][date]=[]
            for voyage in test_voyages[lp][date]:
                voyage_predictions = []
                for idx in range(len(voyage)):
                    current_node = voyage[idx][0]
                    next_node_predicted, model_used = predict_next_node(voyage, idx, edges_count_first[lp],edges_count_second[lp])
                    next_node_true = voyage[idx + 1][0] if idx + 1 < len(voyage) else "End_Node"
                    voyage_predictions.append({
                        'current_node': current_node,
                        'current_node_name':camera_name_dict[current_node] if  current_node in camera_name_dict else None,
                        'predicted_next_node': next_node_predicted,
                        'model_used': model_used,
                        'ground_truth': next_node_true,
                        'predicted_next_node_Name':camera_name_dict[next_node_predicted] if next_node_predicted in camera_name_dict else None
                    })
                if next_node_true==next_node_predicted:
                     precision[lp]['precision']+=1
                     precision['all']['precision']+=1
                precision[lp]['total_number_of_nodes']+=1
                precision['all']['total_number_of_nodes']+=1
                predicted_results[lp][date].append(voyage_predictions)
        precision[lp]['precision']=precision[lp]['precision']/precision[lp]['total_number_of_nodes']
precision['all']['precision']=precision['all']['precision']/precision['all']['total_number_of_nodes']
# Optionally, print or analyze the predictions
for lp in predicted_results:
    print(f"Predictions for {lp}:")
        
    # Get the dates sorted by the length of their corresponding voyage_pred list
    for date in predicted_results[lp]:
         print(f"in{date}:")
         for voyage_pred in predicted_results[lp][date]: 
             for pred in voyage_pred:
                 print(pred)


