import json
from collections import defaultdict

def reverse_json(input_file, output_file):
    # Load the input JSON file into a Python dictionary
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary to hold the reversed data
    reversed_data = defaultdict(lambda: {"parentReceptacles": [], "ObjectState": None})
    
    # Iterate through the original data
    for obj, details in data.items():
        # Handle objects with parent receptacles
        if details["parentReceptacles"]:
            for parent in details["parentReceptacles"]:
                reversed_data[parent]["parentReceptacles"].append(obj)
        # Handle objects with ObjectState
        if details["ObjectState"] is not None:
            reversed_data[obj]["ObjectState"] = details["ObjectState"]
    
    # Create the final scene list
    scene = {}
    
    for parent, details in reversed_data.items():
        # Determine the state of the parent object
        parent_state = details["ObjectState"] if details["ObjectState"] else "clear"
        
        # Create the scene entry for the parent object
        scene[parent] = {
            # "object": parent,
            "state": parent_state,
            "contains": []
        }
        
        for child in details["parentReceptacles"]:
            # Determine the state of the contained object
            child_state = "in" if parent_state == "Closed" else "on"
            
            # Create the entry for the contained object
            child_entry = {
                "object": child,
                "state": child_state
            }

            scene[parent]["contains"].append(child)
        
        # scene.append(scene_entry)
    
    # Prepare the final output dictionary
    # final_output = {"scene": scene}
    
    # Save the final output to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(scene, f, indent=2)

# Specify input and output file paths
input_file = 'filtered_scene_graph.json'
output_file = 'reversed_scene_graph.json'

# Reverse the JSON structure and save to a new file
reverse_json(input_file, output_file)

print(f"Reversed scene graph saved to {output_file}.")
