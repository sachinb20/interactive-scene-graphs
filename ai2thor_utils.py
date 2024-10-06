import numpy as np
import cv2
import math
from sg_utils import reverse_json, filter_sg,create_scene_graph_from_metadata
from utils import closest_position, calculate_object_center
import json
import random
import os
# def get_obj_id(input_key, data):
#     input_key = input_key.lower()
#     matching_keys = []
#     for key in data.keys():
#         if key.lower().startswith(input_key):
#             matching_keys.append(key)
#     return matching_keys[0]

def metadata_scene_graph(controller, scene_graph):
# Create a new scene graph with receptacleObjectIds populated
    metadata_scene_graph = {}

    # Map the receptacleObjectIds from controller.last_event metadata
    for obj in controller.last_event.metadata["objects"]:
        object_id = obj["objectId"]
        receptacle_object_ids = obj.get("receptacleObjectIds", [])
        if obj["openable"]:
            if obj["isOpen"]:
                object_state = "Open"
            else:
                object_state = "Closed"
        else:
            object_state = "clear"
        
        # Copy the data from the original scene graph but replace 'contains' with receptacleObjectIds
        if object_id in scene_graph:
            metadata_scene_graph[object_id] = scene_graph[object_id].copy()
            metadata_scene_graph[object_id]["contains"] = receptacle_object_ids
            metadata_scene_graph[object_id]["State"] = object_state

    return metadata_scene_graph
def disable_objects(controller, scene):
    with open(f'/home/hypatia/Sachin_Workspace/interactive-scene-graphs/sg_data/disabled_objects/{scene}.txt', 'r') as file:
        lines = file.readlines()

    # Loop through each line in the file and call controller.step for each object
    for line in lines:
        # Strip any extra whitespace
        line = line.strip()
        print(line)
        # Call the controller.step function with the formatted action and objectId
        controller.step(
            action="DisableObject",
            objectId=line
        )

        


def get_only_obj_ids(data):
    keys_and_objects = set()
    for key, value in data.items():
        
        if "contains" in value:
            
            objects = set(value["contains"])
            keys_and_objects = keys_and_objects.union(objects)

    return keys_and_objects  

def get_obj_rec_ids(data):
    try:
        # Initialize a set with the keys of the data
        keys_and_objects = set(data.keys())
        
        # Loop through the dictionary
        for key, value in data.items():
            try:
                # Check if "contains" exists in the value (value should be a dictionary)
                if "contains" in value:
                    # Add the objects contained within to the keys_and_objects set
                    objects = set(value["contains"])
                    keys_and_objects = keys_and_objects.union(objects)
            except KeyError:
                print(f"KeyError: 'contains' not found for key {key}")
            except TypeError:
                print(f"TypeError: Expected a dictionary for key {key}, got {type(value)}")

        return keys_and_objects
    except Exception as e:
        print(f"Unexpected error in get_obj_rec_ids: {e}")
        return set()


def get_obj_id(input_key, data):
    try:
        # Retrieve all keys and object IDs
        keys_and_objects = get_obj_rec_ids(data)

        # Normalize the input key to lowercase
        input_key = input_key.lower()
        matching_keys = []
        
        # Check for matching keys
        for key in keys_and_objects:
            if key.lower().startswith(input_key):
                matching_keys.append(key)

        # Ensure at least one match is found, otherwise raise an error
        if not matching_keys:
            raise ValueError(f"No matching keys found for input '{input_key}'")

        return matching_keys[0]
    
    except ValueError as ve:
        print(ve)
        return None  # or handle it as appropriate
    except Exception as e:
        print(f"Unexpected error in get_obj_id: {e}")
        return None


def save_frame(controller,state,folder_name = "action_images/"):

    os.makedirs(folder_name, exist_ok=True)
    bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)
    file_path = os.path.join(folder_name, f"{state}.jpg")
    
    # Save the image
    cv2.imwrite(file_path, bgr_frame)

    return bgr_frame

def shift_indices(arr):
    continuous_parts = []
    current_part = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_part.append(arr[i])
        else:
            continuous_parts.append(current_part)
            current_part = [arr[i]]

    print(current_part)
    print(continuous_parts)
    if continuous_parts == []:
      return current_part
    else:
      return np.concatenate((current_part, continuous_parts[0]))
    
def change_scene(controller):   
    
    # controller.step(
    #     action="OpenObject",
    #     objectId="Fridge|-02.48|+00.00|-00.78",
    #     openness=1,
    #     forceAction=True
    # )

    # controller.step(
    # action="PickupObject",
    # objectId="Apple|-00.48|+00.97|+00.41",
    # forceAction=True,
    # manualInteract=False
    # )

    # controller.step(
    # action="PutObject",
    # objectId="CounterTop|+01.59|+00.95|+00.41",
    # forceAction=True,
    # placeStationary=True
    # )
    
    event = controller.step(
        action="GetSpawnCoordinatesAboveReceptacle",
        objectId="CounterTop|+00.47|+00.95|-01.63",
        anywhere=True  
    )
    print(event.metadata["actionReturn"][0])

    event = controller.step(
    action="PlaceObjectAtPoint",
    objectId="Mug|+01.45|+00.91|-01.23",
    # position={
    #     "x": 1.38,
    #     "y": 00.81,
    #     "z": -1.27
    # }
    position = event.metadata["actionReturn"][0]
    )
    print(event.metadata["lastActionSuccess"])

    scene_graph, object_list = create_scene_graph_from_metadata(event.metadata["objects"])

    # Save scene graph to a file
    file_path = "scene_graph_modified.json"
    with open(file_path, "w") as json_file:
        json.dump(scene_graph, json_file, indent=2)

    print(f"Scene graph saved to {file_path}")

    obj = next(obj for obj in event.metadata["objects"] if obj["objectType"] == "Apple")
    return obj["objectId"]




    
def visible_state(controller,target_receptacle,scene_graph):
    visibility_states = []

    for angle in range(12):
        last_rot = controller.last_event.metadata["agent"]["rotation"]["y"]
        controller.step(
            action="RotateLeft",
            degrees=30
        )
        #In case agent is stuck while rotating
        if last_rot == controller.last_event.metadata["agent"]["rotation"]["y"]:
            print("mera yasu yasu")
            rewind_angle = 1*30
            return rewind_angle
        

        # types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
        # assert target_receptacle in types_in_scene
        # # print(types_in_scene)
        # obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_receptacle)

        obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectId"] == get_obj_id(target_receptacle,scene_graph))


        print(obj['visible'])
        visibility_states.append(obj['visible'])
        save_frame(controller,str(angle+30))
 
    print(visibility_states)

    return visibility_states

def perturb(controller):
    # Define a list of actions
    actions = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
    
    # Choose a random action
    action = random.choice(actions)
    
    # Execute the chosen action
    controller.step(action)
    return None

def rotate_angle(controller,target_receptacle,scene_graph):


    # # Find the indices of all True values in the cyclical array
    # true_indices = [i for i, val in enumerate(visibility_states) if val]

    # # Calculate the total number of True values
    # num_true = len(true_indices)

    # # Calculate the length of the array
    # array_length = len(visibility_states)

    # # Initialize the shifted array
    # shifted_visibility_states = [False] * array_length

    # # Shift the segments if necessary
    # if num_true > 0:
    #     first_true_index = true_indices[0]
    #     shift_amount = array_length - first_true_index
    #     for i, val in enumerate(visibility_states):
    #         if val:
    #             shifted_visibility_states[(i + shift_amount) % array_length] = True

    visibility_states = visible_state(controller,target_receptacle,scene_graph)

    #Not well written but returns 30 degree rewind incase of hitting obs during rotation
    if type(visibility_states)  == int:
        return visibility_states
    
    #check whether not visible then pertube the agent
    i=0
    while all(not elem for elem in visibility_states) and i<1:
        perturb(controller)
        print("#####################33")
        print(target_receptacle)
        visibility_states = visible_state(controller,target_receptacle,scene_graph)
        i+=1

    if all(not elem for elem in visibility_states):
        rewind_angle = None
        return rewind_angle
    
    

    true_indices = [i for i, val in enumerate(visibility_states) if val]

    # Find the indices of all True values in the shifted array
    shifted_true_indices = shift_indices(true_indices)
    midpoint_index = (len(shifted_true_indices) - 1) // 2

    # Get the index of the middle True value
    middle_index = shifted_true_indices[midpoint_index]
    print(middle_index)
    # Calculate the angle needed to rewind the rotation to that position
    rewind_angle = (11-middle_index) * 30




    return rewind_angle



def get_angle_and_closest_position(controller, object_type, scene_graph):
    # Extracting object and agent positions
    # types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    # assert object_type in types_in_scene
    # # print(types_in_scene)
    # obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == object_type)

    object_id = get_obj_id(object_type, scene_graph)
  
    obj_position = calculate_object_center(scene_graph[object_id]['BoundingBox'])

    # Save the reachable positions of the scene to a file
    reachable_positions = controller.step(
        action="GetReachablePositions", raise_for_failure=True
    ).metadata["actionReturn"]

    
    closest = closest_position(obj_position, reachable_positions)

    target_obj = controller.last_event.metadata["objects"][0]
    obj_x = target_obj["position"]["x"]
    obj_z = target_obj["position"]["z"]

    agent_position = controller.last_event.metadata["agent"]["position"]
    agent_x = agent_position["x"]
    agent_z = agent_position["z"]

    delta_x = obj_x - agent_x
    delta_z = obj_z - agent_z
    angle_rad = math.atan2(delta_z, delta_x)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, closest, object_id






