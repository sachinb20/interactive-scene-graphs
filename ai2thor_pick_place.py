import copy
import json
import os
from pathlib import Path
import random
import pickle
import warnings
import cv2
import math
from PIL import Image
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import trange
import argparse
# from lang_sam import LangSAM
from LLM4PicknPlace import PickAndPlaceAgent
from Image_tagging import TaggingModule


import math
from typing import List,Dict
import json
# import open_clip
import torch
import torchvision




def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]):
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes



def create_scene_graph(objects):
    scene_graph = {}
    scene_graph["Agent"]={  
            "position": {
      "x": -0.34021228551864624,
      "y": 0.9094499349594116,
      "z": 1.0938019752502441
    },
        "center": {
      "x": -0.34021228551864624,
      "y": 0.9094499349594116,
      "z": 1.0938019752502441
    },

            "BoundingBox": [
      [
        -2.0334811210632324,
        1.7936310768127441,
        -0.2726714611053467
      ],
      [
        -2.0334811210632324,
        1.7936310768127441,
        -1.2774642705917358
      ],
      [
        -2.0334811210632324,
        -0.008308768272399902,
        -0.2726714611053467
      ],
      [
        -2.0334811210632324,
        -0.008308768272399902,
        -1.2774642705917358
      ],
      [
        -2.7560410499572754,
        1.7936310768127441,
        -0.2726714611053467
      ],
      [
        -2.7560410499572754,
        1.7936310768127441,
        -1.2774642705917358
      ],
      [
        -2.7560410499572754,
        -0.008308768272399902,
        -0.2726714611053467
      ],
      [
        -2.7560410499572754,
        -0.008308768272399902,
        -1.2774642705917358
      ]
    ],
            "parentReceptacles": ["Floor|+00.00|+00.00|+00.00"],
            "ObjectState": None
            }
    

    OBJECT_LIST = []
    for obj in objects:
        obj_id = obj["objectId"]
        aabb = obj["objectOrientedBoundingBox"]["cornerPoints"] if obj["pickupable"] else obj["axisAlignedBoundingBox"]["cornerPoints"]

        if obj["openable"]:
            if obj["isOpen"]:
                object_state = "Open"
            else:
                object_state = "Closed"
        else:
            object_state = None
        
        scene_graph[obj_id] = {
            "position": obj["position"],
            "center": obj["axisAlignedBoundingBox"]["center"],
            "BoundingBox": aabb,
            "parentReceptacles": obj["parentReceptacles"],
            "ObjectState": object_state
        }
        OBJECT_LIST.append(obj["objectType"])


    return scene_graph, OBJECT_LIST

def update_scene_graph(scene_graph,action,obj_id,recept_id):

    if action == "Pickup":
        scene_graph[obj_id]['parentReceptacles'] = ["Agent"]

    elif action == "Putdown":
        scene_graph[obj_id]['parentReceptacles'] = [recept_id]

    elif action == "Open":
        scene_graph[obj_id]['ObjectState'] = "Open"

    elif action == "Close":
        scene_graph[obj_id]['ObjectState'] = "Close"

    elif action == "Navigate":
        scene_graph = scene_graph

    return scene_graph

def open_receptacle(scene_graph,controller,obj_id,action_no):

    controller.step(
        action="OpenObject",
        objectId=obj_id,
        openness=1,
        forceAction=False
    )

    save_frame(controller,str(action_no))

    #Verify Action 

    # #Update SG
    action = "Open"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph

def pick_object(scene_graph,controller,obj_id,action_no):

    event = controller.step(
    action="PickupObject",
    objectId=obj_id,
    forceAction=False,
    manualInteract=False
    )

    bgr_frame = save_frame(controller,str(action_no))

    #Verify Action


    #Update SG
    action = "Pickup"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph

def put_object(scene_graph,controller,obj_id,action_no):

    controller.step(
    action="PutObject",
    objectId=obj_id,
    forceAction=False,
    placeStationary=True
    )

    save_frame(controller,str(action_no))


    #Verify
    action = "Putdown"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,obj_id)    

    return scene_graph

def close_receptacle(scene_graph,controller,obj_id,action_no):

    controller.step(
    action="CloseObject",
    objectId=obj_id,
    forceAction=False
    )
    
    save_frame(controller,str(action_no))
    
    # #Verify

    # #Update SG
    action = "Close"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph

def navigate(controller,object,scene_graph,action_no,tagging_module):

    angle_deg, closest, obj_id = get_angle_and_closest_position(controller,object,scene_graph)
    controller.step(action="Teleport", **closest)
    angle = rotate_angle(controller, object)
    controller.step(
        action="RotateRight",
        degrees=angle
    )


    bgr_frame = save_frame(controller,str(action_no))


    closest_items = find_closest_items(controller.last_event.metadata["agent"]["position"], scene_graph, num_items=10)
    caption, text_prompt = tagging_module.predict(bgr_frame)
    print(text_prompt)
    print(closest_items)

    return obj_id, scene_graph

    
def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    out = reachable_positions[0]
    min_distance = float('inf')
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out

def find_keys(input_key, data):
    input_key = input_key.lower()
    matching_keys = []
    for key in data.keys():
        if key.lower().startswith(input_key):
            matching_keys.append(key)
    return matching_keys

def get_angle_and_closest_position(controller, object_type, scene_graph):
    # Extracting object and agent positions
    # types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    # assert object_type in types_in_scene
    # # print(types_in_scene)
    # obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == object_type)

    keys = find_keys(object_type, scene_graph)
    object_id = keys[0]                       #Choose first key
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



def euclidean_distance(pos1, pos2):
    # print(pos1)
    # print(pos2)
    return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2 + (pos1['z'] - pos2['z'])**2)

def calculate_object_center(bounding_box):
    x_coords = [point[0] for point in bounding_box]
    y_coords = [point[1] for point in bounding_box]
    z_coords = [point[2] for point in bounding_box]
    center = {
        'x': sum(x_coords) / len(bounding_box),
        'y': sum(y_coords) / len(bounding_box),
        'z': sum(z_coords) / len(bounding_box)
    }
    return center

def find_closest_items(agent_position, scene_graph, num_items=5):
    distances = {}
    for obj_id, obj_data in scene_graph.items():
        # obj_position = obj_data['position']
        obj_aabb = obj_data['BoundingBox']
        obj_center = calculate_object_center(obj_aabb)
        # Adjusting object position relative to the agent
        obj_position_global = {
            'x': obj_center['x'] ,
            'y': obj_center['y'] ,
            'z': obj_center['z'] 
        }
        # Calculate distance from agent to object
        distance = euclidean_distance(agent_position, obj_position_global)
        distances[obj_id] = distance
    # Sort distances and return the closest items
    closest_items = sorted(distances.items(), key=lambda x: x[1])[:num_items]
    return closest_items



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
    
def visible_state(controller,target_receptacle):
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
        

        types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
        assert target_receptacle in types_in_scene
        # print(types_in_scene)
        obj = next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == target_receptacle)
        print(obj['visible'])
        visibility_states.append(obj['visible'])

        save_frame(controller,target_receptacle+'/'+str(angle+30))
 
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

def rotate_angle(controller,target_receptacle):


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

    visibility_states = visible_state(controller,target_receptacle)

    #Not well written but returns 30 degree rewind incase of hitting obs during rotation
    if type(visibility_states)  == int:
        return visibility_states
    
    #check whether not visible then pertube the agent
    while all(not elem for elem in visibility_states):
        perturb(controller)
        visibility_states = visible_state(controller,target_receptacle)


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

def main(args: argparse.Namespace):
    # save_folder_name = (
    #     args.scene_name
    #     if args.save_suffix is None
    #     else args.scene_name + "_" + args.save_suffix
    # )
    # save_root = args.dataset_root + "/" + save_folder_name + "/"
    # os.makedirs(save_root, exist_ok=True)

    # args.save_folder_name = save_folder_name
    # args.save_root = save_root

    # Initialize the controller
    controller = Controller(
        	# agentMode = "arm",
        agentMode="default",
        visibilityDistance=1.5,
        # scene=get_scene(args.scene_name),
        scene="FloorPlan6",
        # step sizes
        gridSize=args.grid_size,
        snapToGrid=False,
        rotateStepDegrees=30,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        # camera properties
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,
    )


    #Create OG SG
    controller.step("MoveBack")
    event = controller.step("MoveAhead")
    scene_graph, object_list = create_scene_graph(event.metadata["objects"])

    # Save scene graph to a file
    file_path = "scene_graph.json"
    with open(file_path, "w") as json_file:
        json.dump(scene_graph, json_file, indent=2)

    print(f"Scene graph saved to {file_path}")


    #Language Query to Decide Planning (LLM Planner) 
    #Eg: Pick the Tomato from the sink and place it in the Fridge

    #TODO: check https://ai2thor.allenai.org/manipulathor/documentation/#interaction
    #TODO: Gaussain Splatting for memory update
    #TODO: https://openreview.net/forum?id=eJhc_CPXQIT

    prompt = "I'd like to have my apple chilled. Could you find a cool place to keep it."
    agent = PickAndPlaceAgent()
    planner = agent.pick_and_place(object_list,prompt)

    tagging_module = TaggingModule()


    print(planner)
    target_receptacle = planner["target_receptacle"]
    object = planner["source_object"]

    # target_receptacle = "Fridge"
    # object = "Apple"

    source_receptacle = scene_graph[find_keys(object, scene_graph)[0]]["parentReceptacles"]

    print(target_receptacle)
    print(source_receptacle)
    print(object)


###############################################################################################
    #Navigate + Tune Location (To View Object + Effective Manip)
    # angle_deg, closest, target_recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
    # event = controller.step(action="Teleport", **closest) 
    # angle = rotate_angle(controller, target_receptacle)
    # # # Rewind the rotation
    # controller.step(
    #     action="RotateRight",  # Rewind the rotation by rotating right
    #     degrees=angle
    # )

    # event = controller.step("MoveBack")
    # bgr_frame=save_frame(controller,"1")


    # #Verify
    # closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
    # caption, text_prompt = tagging_module.predict(bgr_frame)
    # print(text_prompt)
    # print(closest_items)   

    target_recept_id, scene_graph = navigate(controller,target_receptacle,scene_graph,1,tagging_module)
    #TODO: Find a score



    # #TargetReceptacle Manipulation (Open)
    # controller.step(
    #     action="OpenObject",
    #     objectId=target_recept_id,
    #     openness=1,
    #     forceAction=False
    # )

    # save_frame(controller,"2")

    # #Verify Action + Update SG

    # # #Update SG
    # action = "Open"
    # scene_graph = update_scene_graph(scene_graph,action,target_recept_id,None)
    scene_graph = open_receptacle(scene_graph,controller,target_recept_id,2)


##########################################################################################
    #
    #SourceReceptaple Manipulation (Open)
    #
########################################################################################3

    #Navigate to Object
    # angle_deg, closest, obj_id = get_angle_and_closest_position(controller,object,scene_graph)
    # controller.step(action="Teleport", **closest)
    # angle = rotate_angle(controller, object)
    # controller.step(
    #     action="RotateRight",
    #     degrees=angle
    # )


    # bgr_frame = save_frame(controller,"3")


    # closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=10)
    # caption, text_prompt = tagging_module.predict(bgr_frame)
    # print(text_prompt)
    # print(closest_items)

    obj_id, scene_graph = navigate(controller,object,scene_graph,3,tagging_module)



    # #Object Pickup
    # event = controller.step(
    # action="PickupObject",
    # objectId=obj_id,
    # forceAction=False,
    # manualInteract=False
    # )

    # bgr_frame = save_frame(controller,"4")



    # #Verify 
    # # black_image = get_mask_with_pointprompt(bgr_frame)
    # # frame = cv2.cvtColor(black_image,cv2.COLOR_RGB2BGR)

    # #Update SG
    # action = "Pickup"
    # scene_graph = update_scene_graph(scene_graph,action,obj_id,None)
    # print(scene_graph[obj_id])

    # print(event.metadata["agent"]["position"])
    scene_graph = pick_object(scene_graph,controller,obj_id,4)



############################################################################
    #Receptacle Navigation

    # angle_deg, closest, recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
    # event = controller.step(action="Teleport", **closest)  
    # angle = rotate_angle(controller, target_receptacle)
    # controller.step(
    #     action="RotateRight",  # Rewind the rotation by rotating right
    #     degrees=angle
    # )


    # bgr_frame = save_frame(controller,"5")
    # closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
    # caption, text_prompt = tagging_module.predict(bgr_frame)
    # print(text_prompt)
    # print(closest_items)
    recept_id, scene_graph = navigate(controller,target_receptacle,scene_graph,5,tagging_module)



#     #Object Putdown
#     controller.step(
#     action="PutObject",
#     objectId=recept_id,
#     forceAction=False,
#     placeStationary=True
# )

#     save_frame(controller,"6")


#     #Verify
#     action = "Putdown"
#     scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    
#     print(scene_graph[obj_id])
    scene_graph = put_object(scene_graph,controller,recept_id,6)
###########################################################################3
    #
    #TargetReceptacle Manipulation (Close)

    # controller.step(
    # action="CloseObject",
    # objectId=recept_id,
    # forceAction=False
    # )
    
    # save_frame(controller,"7")
    
    # # #Verify

    # # #Update SG
    # action = "Close"
    # scene_graph = update_scene_graph(scene_graph,action,recept_id,None)
    scene_graph = close_receptacle(scene_graph,controller,recept_id,7)
############################################################################3

    #
    #SourceReceptaple Manipulation (Close)
    #

##########################################################################



    with open(file_path, "w") as json_file:
        json.dump(scene_graph, json_file, indent=2)

    print(f"Scene graph saved to {file_path}")



    print("#########################################################")
    print("Task Completed")
    print("##########################################################")



            
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Program Arguments")
    parser.add_argument(
        "--dataset_root",
        default=str(Path("~/ldata/ai2thor/").expanduser()),
        help="The root path to the dataset.",
    )
    parser.add_argument(
        "--grid_size",
        default=0.5,
        type=float,
        help="The translational step size in the scene (default 0.25).",
    )
    
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--interact", action="store_true", help="Run in interactive mode. Requires GUI access."
    )
    parser.add_argument(
        "--traj_file_name", type=str, default="trajectory.json", 
        help="The name of the trajectory file to load."
    )
    
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save trajectories from the interaction."
    )
    parser.add_argument(
        "--height", default=480, type=int, help="The height of the image."
    )
    parser.add_argument(
        "--width", default=640, type=int, help="The width of the image."
    )
    parser.add_argument(
        "--fov", default=90, type=int, help="The (vertical) field of view of the camera."
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save the video of the generated RGB frames."
    )
    
    parser.add_argument("--scene_name", default="train_3")
    parser.add_argument("--save_suffix", default=None)
    parser.add_argument("--randomize_lighting", action="store_true")
    parser.add_argument("--randomize_material", action="store_true")

    # Randomly remove objects in the scene
    parser.add_argument(
        "--randomize_remove_ratio",
        default=0.0,
        type=float,
        help="The probability to remove any object in the scene (0.0 - 1.0)",
    )
    parser.add_argument(
        "--randomize_remove_level", 
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="""What kind of objects to remove duing randomization 
        1: all objects except those in NOT_TO_REMOVE; 
        2: objects that are pickupable or moveable;
        3: objects that are pickupable""",
    )
    
    # Randomly moving objects in the scene
    parser.add_argument(
        "--randomize_move_pickupable_ratio",
        default=0.0,
        type=float,
        help="The ratio of pickupable objects to move.",
    )
    parser.add_argument(
        "--randomize_move_moveable_ratio",
        default=0.0,
        type=float,
        help="The ratio of moveable objects to move.",
    )
    
    parser.add_argument(
        "--topdown_only", action="store_true", help="Generate and save only the topdown view."
    )
    parser.add_argument(
        "--depth_scale", default=1000.0, type=float, help="The scale of the depth."
    )
    parser.add_argument(
        "--n_sample",
        default=-1,
        type=int,
        help="The number of images to generate. (-1 means all reachable positions are sampled)",
    )
    parser.add_argument(
        "--sample_method",
        default="uniform",
        choices=["random", "uniform", "from_file"],
        help="The method to sample the poses (random, uniform, from_file).",
    )
    parser.add_argument("--seed", default=0, type=int)
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.interact:
        main_interact(args)
    else:
        main(args)
