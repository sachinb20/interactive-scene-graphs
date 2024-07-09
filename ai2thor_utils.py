import numpy as np
import cv2
import math
from isg import create_scene_graph, update_scene_graph
from utils import closest_position, calculate_object_center
import json
import random

def get_obj_id(input_key, data):
    input_key = input_key.lower()
    matching_keys = []
    for key in data.keys():
        if key.lower().startswith(input_key):
            matching_keys.append(key)
    return matching_keys[0]

def save_frame(controller,state):

    bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite("action_images/"+state+'.jpg', bgr_frame)

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

    scene_graph, object_list = create_scene_graph(event.metadata["objects"])

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






