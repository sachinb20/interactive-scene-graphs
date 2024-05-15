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


import math
from typing import List,Dict
import json
# import open_clip
import torch
import torchvision

# from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption

# try: 
#     from groundingdino.util.inference import Model
#     from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# except ImportError as e:
#     print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
#     raise e

# Set up some path used in this script
# Assuming all checkpoint files are downloaded as instructed by the original GSA repo
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    
import sys
TAG2TEXT_PATH = os.path.join(GSA_PATH, "Tag2Text")
# EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
RAM_PATH = os.path.join(GSA_PATH, "recognize-anything")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
# sys.path.append(EFFICIENTSAM_PATH)
sys.path.append(RAM_PATH)
try:
    from ram.models.tag2text import tag2text
    from ram.models.ram import ram as rm
    from ram import inference
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# # GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# # Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")



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


# def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
#     backup_image = image.copy()
    
#     image = Image.fromarray(image)
    
#     # padding = args.clip_padding  # Adjust the padding amount as needed
#     padding = 20  # Adjust the padding amount as needed
    
#     image_crops = []
#     image_feats = []
#     text_feats = []

    
#     for idx in range(len(detections.xyxy)):
#         # Get the crop of the mask with padding
#         x_min, y_min, x_max, y_max = detections.xyxy[idx]

#         # Check and adjust padding to avoid going beyond the image borders
#         image_width, image_height = image.size
#         left_padding = min(padding, x_min)
#         top_padding = min(padding, y_min)
#         right_padding = min(padding, image_width - x_max)
#         bottom_padding = min(padding, image_height - y_max)

#         # Apply the adjusted padding
#         x_min -= left_padding
#         y_min -= top_padding
#         x_max += right_padding
#         y_max += bottom_padding

#         cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
#         # Get the preprocessed image for clip from the crop 
#         preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

#         crop_feat = clip_model.encode_image(preprocessed_image)
#         crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
#         class_id = detections.class_id[idx]
#         tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
#         text_feat = clip_model.encode_text(tokenized_text)
#         text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
#         crop_feat = crop_feat.cpu().numpy()
#         text_feat = text_feat.cpu().numpy()

#         image_crops.append(cropped_image)
#         image_feats.append(crop_feat)
#         text_feats.append(text_feat)
        
#     # turn the list of feats into np matrices
#     image_feats = np.concatenate(image_feats, axis=0)
#     text_feats = np.concatenate(text_feats, axis=0)

#     return image_crops, image_feats, text_feats

# def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)

# def get_sam_segmentation_from_point_and_box(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray,input_point: np.ndarray,input_label: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, _ = sam_predictor.predict(
#                 point_coords=input_point,
#                 point_labels=input_label,
#                 box=box,
#                 multimask_output=True,
#             )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)


# def get_sam_predictor(variant: str, device: str | int) -> SamPredictor:
#     if variant == "sam":
#         sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
#         sam.to(device)
#         sam_predictor = SamPredictor(sam)
#         return sam_predictor
    
#     if variant == "mobilesam":
#         from MobileSAM.setup_mobile_sam import setup_model
#         MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
#         checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
#         mobile_sam = setup_model()
#         mobile_sam.load_state_dict(checkpoint, strict=True)
#         mobile_sam.to(device=device)
        
#         sam_predictor = SamPredictor(mobile_sam)
#         return sam_predictor

#     elif variant == "lighthqsam":
#         from LightHQSAM.setup_light_hqsam import setup_model
#         HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
#         checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
#         light_hqsam = setup_model()
#         light_hqsam.load_state_dict(checkpoint, strict=True)
#         light_hqsam.to(device=device)
        
#         sam_predictor = SamPredictor(light_hqsam)
#         return sam_predictor
        
#     elif variant == "fastsam":
#         raise NotImplementedError
#     else:
#         raise NotImplementedError
    

# def get_mask(bgr_frame):

#     image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
#     sam_variant = "sam"
#     sam_predictor = get_sam_predictor(sam_variant, args.device)
#     mask = get_sam_segmentation_from_xyxy(
#                 sam_predictor=sam_predictor,
#                 image=image_rgb,
#                 xyxy=np.array([[260,280,380,400]])
#         )
    
#     # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # # Create a white background
#     # white_bg = np.ones_like(bgr_frame) * 255

#     # # Draw contours on the white background
#     # cv2.drawContours(white_bg, contours, -1, (0, 0, 0), thickness=1)

#     # # Save the image
#     # cv2.imwrite("masked.jpg", white_bg)

#     image_np = np.array(image_rgb)
#     print(mask.shape)
#     print(np.shape(image_np))
#     mask = mask[0]
#     print(mask.shape)
#     # Create an all-black image of the same shape as the input image
#     black_image = np.zeros_like(image_np)
#     # Wherever the mask is True, replace the black image pixel with the original image pixel
#     black_image[mask] = image_np[mask]
#     # convert back to pil image
#     black_image = Image.fromarray(black_image)

#     black_image.save("saved_image.png")

#     return None    

# def get_mask_with_pointprompt(bgr_frame):

#     image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
#     sam_variant = "sam"
#     sam_predictor = get_sam_predictor(sam_variant, args.device)
#     mask = get_sam_segmentation_from_point_and_box(
#                 sam_predictor=sam_predictor,
#                 image=image_rgb,
#                 xyxy=np.array([[230,240,410,480]]),
#                 input_point = np.array([[320, 340]]),
#                 input_label = np.array([1])
#         )
    
#     # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # # Create a white background
#     # white_bg = np.ones_like(bgr_frame) * 255

#     # # Draw contours on the white background
#     # cv2.drawContours(white_bg, contours, -1, (0, 0, 0), thickness=1)

#     # # Save the image
#     # cv2.imwrite("masked.jpg", white_bg)

#     image_np = np.array(image_rgb)
#     print(mask.shape)
#     print(np.shape(image_np))
#     mask = mask[0]
#     print(mask.shape)
#     # Create an all-black image of the same shape as the input image
#     black_image = np.zeros_like(image_np)
#     # Wherever the mask is True, replace the black image pixel with the original image pixel
#     black_image[mask] = image_np[mask]
#     # convert back to pil image
#     # black_image = Image.fromarray(black_image)

#     # black_image.save("pepper_shaker_masked.png")
#     cv2.imwrite("pepper_shaker_masked.png",black_image)
#     return black_image  

def tagging_module(bgr_frame):

    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)

    specified_tags='None'
    # load model
    tagging_model = tag2text(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                            image_size=384,
                                            vit='swin_b',
                                            delete_tag_index=delete_tag_index)
    # threshold for tagging
    # we reduce the threshold to obtain more tags
    tagging_model.threshold = 0.64 
    tagging_model = tagging_model.eval().to(args.device)
    tagging_transform = TS.Compose([
                            TS.Resize((384, 384)),
                            TS.ToTensor(), 
                            TS.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                                     ])
    
    image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Convert to RGB color space
    image_pil = Image.fromarray(image_rgb)
    raw_image = image_pil.resize((384, 384))
    raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)

    res = inference.inference_tag2text(raw_image , tagging_model, specified_tags)
    caption=res[2]
    
    text_prompt=res[0].replace(' |', ',')

    return caption, text_prompt

def save_frame(controller,state):

    bgr_frame = cv2.cvtColor(controller.last_event.frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(state+'.jpg', bgr_frame)

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
    angle_deg, closest, target_recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
    event = controller.step(action="Teleport", **closest) 
    angle = rotate_angle(controller, target_receptacle)
    # # Rewind the rotation
    controller.step(
        action="RotateRight",  # Rewind the rotation by rotating right
        degrees=angle
    )

    event = controller.step("MoveBack")
    bgr_frame=save_frame(controller,"1")


    #Verify
    closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
    caption, text_prompt = tagging_module(bgr_frame)
    print(text_prompt)
    print(closest_items)   



    # #TargetReceptacle Manipulation (Open)
    controller.step(
        action="OpenObject",
        objectId=target_recept_id,
        openness=1,
        forceAction=False
    )

    save_frame(controller,"2")

    #Verify Action + Update SG

    # #Update SG
    action = "Open"
    scene_graph = update_scene_graph(scene_graph,action,target_recept_id,None)


##########################################################################################
    #
    #SourceReceptaple Manipulation (Open)
    #
########################################################################################3

    #Navigate to Object
    angle_deg, closest, obj_id = get_angle_and_closest_position(controller,object,scene_graph)
    controller.step(action="Teleport", **closest)
    angle = rotate_angle(controller, object)
    controller.step(
        action="RotateRight",
        degrees=angle
    )


    bgr_frame = save_frame(controller,"3")


    closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
    caption, text_prompt = tagging_module(bgr_frame)
    print(text_prompt)
    print(closest_items)




    #Object Pickup
    event = controller.step(
    action="PickupObject",
    objectId=obj_id,
    forceAction=False,
    manualInteract=False
    )

    bgr_frame = save_frame(controller,"4")



    #Verify 
    # black_image = get_mask_with_pointprompt(bgr_frame)
    # frame = cv2.cvtColor(black_image,cv2.COLOR_RGB2BGR)

    #Update SG
    action = "Pickup"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)
    print(scene_graph[obj_id])

    print(event.metadata["agent"]["position"])



############################################################################
    #Receptacle Navigation

    angle_deg, closest, recept_id = get_angle_and_closest_position(controller,target_receptacle,scene_graph)
    event = controller.step(action="Teleport", **closest)  
    angle = rotate_angle(controller, target_receptacle)
    controller.step(
        action="RotateRight",  # Rewind the rotation by rotating right
        degrees=angle
    )


    bgr_frame = save_frame(controller,"5")
    closest_items = find_closest_items(event.metadata["agent"]["position"], scene_graph, num_items=5)
    caption, text_prompt = tagging_module(bgr_frame)
    print(text_prompt)
    print(closest_items)



    #Object Putdown
    controller.step(
    action="PutObject",
    objectId=recept_id,
    forceAction=False,
    placeStationary=True
)

    save_frame(controller,"6")


    #Verify
    action = "Putdown"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    
    print(scene_graph[obj_id])
###########################################################################3
    #
    #TargetReceptacle Manipulation (Close)

    controller.step(
    action="CloseObject",
    objectId=recept_id,
    forceAction=False
    )
    
    save_frame(controller,"7")
    
    # #Verify

    # #Update SG
    action = "Close"
    scene_graph = update_scene_graph(scene_graph,action,recept_id,None)
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
