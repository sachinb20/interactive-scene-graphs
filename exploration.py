import copy
import json
import os
from pathlib import Path

from PIL import Image
# import imageio
import matplotlib.pyplot as plt
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import trange
import argparse
# from lang_sam import LangSAM

from Image_tagging import RAM

import math
from typing import List,Dict
import json
import open_clip


import copy

from ai2thor_actions import open_receptacle, close_receptacle,crouch_n_image, pick_object, put_object,navigate,crouch,stand,camera_rotate
from sg_utils import reverse_json, filter_sg,create_scene_graph_from_metadata
from ai2thor_utils import get_only_obj_ids,disable_objects
from Image_tagging import Detic
import pdb 


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

def instances_to_dict(instances):
    # Initialize an empty dictionary to store the data
    instances_dict = {}

    # Iterate over the fields in the Instances object
    for field in instances.get_fields().keys():
        # Convert each field to a list or tensor
        instances_dict[field] = instances.get(field).cpu().numpy() if instances.get(field).is_cuda else instances.get(field).numpy()

    return instances_dict

def save_dict_to_file(data_dict, folder_name, file_name):
    # Ensure the folder exists, create it if it doesn't
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Define the full path for the file
    file_path = os.path.join(folder_name, file_name)
    
    # Save the dictionary as a JSON file
    with open(file_path, 'w') as file:
        json.dump(data_dict, file, indent=4)

def save_clip_features(image_directory,output_directory,name):

    image_path = os.path.join(image_directory, name+'.jpg')
    image1 = preprocess(Image.open(image_path)).unsqueeze(0)
    image_features1 = model.encode_image(image1)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)

    np.save(output_directory+name+'.npy', image_features1.cpu().detach().numpy())




def main():
    # save_folder_name = (
    #     args.scene_name
    #     if args.save_suffix is None
    #     else args.scene_name + "_" + args.save_suffix
    # )
    # save_root = args.dataset_root + "/" + save_folder_name + "/"
    # os.makedirs(save_root, exist_ok=True)

    # args.save_folder_name = save_folder_name
    # args.save_root = save_root
    scene = "FloorPlan_Val3_4"
    # Initialize the controller
    controller = Controller(
        	# agentMode = "arm",
        agentMode="default",
        visibilityDistance=1.5,
        # scene=get_scene(args.scene_name),
        scene=scene,
        # step sizes
        gridSize=0.1,
        snapToGrid=False,
        rotateStepDegrees=30,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        # camera properties
        width=320,
        height=480,
        fieldOfView=90,
        # platform=CloudRendering,
    )


    disable_objects(controller,scene)

    #Create OG SG
    controller.step("MoveBack")
    event = controller.step("MoveAhead")

    path = "/home/hypatia/Sachin_Workspace/interactive-scene-graphs"
    action_images = f"action_images/{scene}/"
    scene_graphs = "scene_graphs/"

    action_images_path = os.path.join(path,action_images)
    scene_graphs_path =  os.path.join(path,scene_graphs)

    if not os.path.exists(action_images_path):
        os.mkdir(action_images_path)
    
    if not os.path.exists(scene_graphs_path):
        os.mkdir(scene_graphs_path)     

    with open(f"/home/hypatia/Sachin_Workspace/interactive-scene-graphs/sg_data/{scene}_sg.json", 'r') as file:
        scene_graph = json.load(file)
    scene_graph_orig = copy.deepcopy(scene_graph)  


    # Directory containing the images
    image_directory = 'exploration/'+scene+'/'
    output_directory = 'exploration/'+scene+'/'+'output/'
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
 

    # file_path = 'objects.txt'
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    # objects = ','.join(line.strip() for line in lines)

    objects = list(get_only_obj_ids(scene_graph))
    
    

    print(scene_graph.keys())
    receptacles = list(scene_graph.keys())


    print(receptacles)

    step = 0
    disable_objects_list = []
    tag = Detic(confidence_threshold=0.3)
    tagging_module = None
    detect_objects = None
    update_sg = False
     

    for recept in receptacles:
        # name = recept.split('|')[0]
        name = recept
        navigate(scene_graph, controller ,recept, name, tagging_module, detect_objects, update_sg, image_directory)
        


        if scene_graph[recept]["state"] == "Closed":
            open_receptacle(scene_graph, controller, recept,name+"_open" ,tagging_module, detect_objects, update_sg,image_directory)
            
            #encode open state:
            save_clip_features(image_directory,output_directory,name+"_open")


            image_path = os.path.join(image_directory, name+"_open"+'.jpg')
            output_path = os.path.join(output_directory, f"output_{name}.jpg")

            object_predictions, vis_output = tag.tag("custom", ",".join([item.split("|")[0] for item in objects]), image_path)         
            predictions = [objects[i] for i in object_predictions['instances'].pred_classes.tolist()]
            intersection = list(set(predictions) & set(scene_graph[recept]["contains"]))

            disable_objects_list.append(list(set(scene_graph[recept]["contains"]) - set(predictions)))
            scene_graph[recept]["contains"] = intersection

            print(intersection)
            vis_output.save(output_path)  
            #Iou_label + clip vector of iou

            #Add to Sg



            close_receptacle(scene_graph, controller, recept, name+"_close",tagging_module, detect_objects, update_sg,image_directory)
            #encode close state:
            save_clip_features(image_directory,output_directory,name+"_close")



        elif scene_graph[recept]["state"] == "clear":
            #clip vector of state
            save_clip_features(image_directory,output_directory,name)

            image_path = os.path.join(image_directory, name+'.jpg')
            output_path = os.path.join(output_directory, f"output_{name}.jpg")

            object_predictions, vis_output = tag.tag("custom", ",".join([item.split("|")[0] for item in objects]), image_path)         
            predictions = [objects[i] for i in object_predictions['instances'].pred_classes.tolist()]
            intersection = list(set(predictions) & set(scene_graph[recept]["contains"]))

            disable_objects_list.append(list(set(scene_graph[recept]["contains"]) - set(predictions)))
            scene_graph[recept]["contains"] = intersection

            print(intersection)
            vis_output.save(output_path)  

            #Iou_label + clip vector of iou

            #Add to Sg
            continue

    print(disable_objects_list)

    with open(f"/home/hypatia/Sachin_Workspace/interactive-scene-graphs/sg_data/{scene}_sg.json", 'w') as file:
        json.dump(scene_graph, file, indent=4)



if __name__ == "__main__":

    main()


