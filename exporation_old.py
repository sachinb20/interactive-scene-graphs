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

from Image_tagging import RAM

import math
from typing import List,Dict
import json
import open_clip
import torch
import torchvision

import copy

from ai2thor_actions import open_receptacle, close_receptacle,crouch_n_image, pick_object, put_object,navigate,crouch,stand,camera_rotate
from sg_utils import reverse_json, filter_sg,create_scene_graph_from_metadata
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
    scene = "FloorPlan6"
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
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,
    )


    #Create OG SG
    controller.step("MoveBack")
    event = controller.step("MoveAhead")

    path = "/home/hypatia/Sachin_Workspace/interactive-scene-graphs"
    action_images = "action_images"
    scene_graphs = "scene_graphs/"

    action_images_path = os.path.join(path,action_images)
    scene_graphs_path =  os.path.join(path,scene_graphs)

    if not os.path.exists(action_images_path):
        os.mkdir(action_images_path)
    
    if not os.path.exists(scene_graphs_path):
        os.mkdir(scene_graphs_path)

    # scene_graph, object_list = create_scene_graph_from_metadata(event.metadata["objects"])
    # scene_graph_orig = copy.deepcopy(scene_graph)       

    with open("/home/hypatia/Sachin_Workspace/interactive-scene-graphs/scene_graphs/FloorPlan6_test.json", 'r') as file:
        scene_graph = json.load(file)
    scene_graph_orig = copy.deepcopy(scene_graph)  


    # Directory containing the images
    image_directory = 'exploration_images'
    output_directory = 'exploration_images/output_images'
    os.makedirs(image_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
 

    file_path = 'objects.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
    objects = ','.join(line.strip() for line in lines)


    # tagging_module = RAM()
    # query = "Describe the central object in the image."
    # image_file = f"action_images/1.jpg"
    # op = tagging_module.tag(query,image_file)
    # print(op)
    # updated_obj_id = change_scene(controller)
    # print(updated_obj_id)



    #Run LLM
    

    print(scene_graph.keys())
    receptacles = list(scene_graph.keys())


    print(receptacles)

    step = 0



     

    for recept in receptacles:
        name = recept.split('|')[0]
        navigate(scene_graph, controller ,recept, name,None,"exploration_images/")
        
        #DEtic masks for up and crouched of receptacles
        tag = Detic(confidence_threshold=0.5)

        image_path = os.path.join(image_directory, name+'.jpg')
        output_path = os.path.join(output_directory, f"output_{name}.jpg")
        predictions, vis_output = tag.tag("custom", name, image_path)         
        vis_output.save(output_path)       

        # instances_dict = instances_to_dict(predictions['instances'])
        # save_dict_to_file(instances_dict, output_directory, name+'.json')



        crouch_n_image(controller,name+"_crouched","exploration_images/")

        image_path = os.path.join(image_directory, name+"_crouched"+'.jpg')
        output_path = os.path.join(output_directory, f"output_{name}_crouched.jpg")
        predictions, vis_output = tag.tag("custom", name, image_path)         
        vis_output.save(output_path)       
        #preprocess to save only instance with most masked area 

        # instances_dict = instances_to_dict(predictions['instances'])
        # save_dict_to_file(instances_dict, output_directory, name+'.json')

        if scene_graph[recept]["state"] == "Closed":
            open_receptacle(scene_graph, controller, recept,name+"_open" ,None,"exploration_images/")
            #encode open state:
            image_path = os.path.join(image_directory, name+"_open"+'.jpg')
            image1 = preprocess(Image.open(image_path)).unsqueeze(0)
            image_features1 = model.encode_image(image1)
            image_features1 /= image_features1.norm(dim=-1, keepdim=True)

            #Detic_custom_mask:
            #Detic_lvis_mask:
            #Iou_label + clip vector of iou
            image_path = os.path.join(image_directory, name+"_open"+'.jpg')
            output_path = os.path.join(output_directory, f"output_{name}_open.jpg")
            predictions, vis_output = tag.tag("custom", objects, image_path)         
            vis_output.save(output_path)  
            
            print(predictions.keys())
            print(predictions)
            # pdb.set_trace() 
            # save_dict_to_file(predictions, output_directory, name+"_open"+'.json')




            crouch_n_image(controller,name+"_open"+"_crouched","exploration_images/")
            #Detic_custom_mask:
            #Detic_lvis_mask:
            #Iou_label + clip vector of iou

            image_path = os.path.join(image_directory, name+"_open_crouched"+'.jpg')
            output_path = os.path.join(output_directory, f"output_{name}_open_crouched.jpg")
            predictions, vis_output = tag.tag("custom", objects, image_path)         
            vis_output.save(output_path)  

            #Add to sg



            close_receptacle(scene_graph, controller, recept,name+"_close",None,"exploration_images/")
            #encode close state:
            image_path = os.path.join(image_directory, name+"_close"+'.jpg')
            image1 = preprocess(Image.open(image_path)).unsqueeze(0)
            image_features1 = model.encode_image(image1)
            image_features1 /= image_features1.norm(dim=-1, keepdim=True)


            
            # crouch_n_image(controller,name+"_crouched"+"_close","exploration_images/")

        # if scene_graph[recept]["state"] == "Open":
        #     close_receptacle(scene_graph, controller, recept,name+"_close",None,"exploration_images/")
        #     #encode close state:
        #     crouch_n_image(controller,name+"_crouched"+"_close","exploration_images/")



        #     open_receptacle(scene_graph, controller, recept,name+"_open",None,"exploration_images/")
        #     #encode open state:
        #     crouch_n_image(controller,name+"_crouched"+"_open","exploration_images/")



        elif scene_graph[recept]["state"] == "clear":

            image_path = os.path.join(image_directory, name+'.jpg')
            output_path = os.path.join(output_directory, f"output_{name}_clear.jpg")
            predictions, vis_output = tag.tag("custom", objects, image_path)         
            vis_output.save(output_path)  

            #Detic_custom_mask:
            #Detic_lvis_mask:
            #Iou_label + clip vector of iou

            #Add to Sg
            continue

# write plan to explore and extract images
# apply detic to receptacles
# apply detic to objects and find iou to make sg
# take clip ft of open,close and of objects


    # llm_plan = [navigate,open_receptacle,navigate,pick_object,navigate,put_object,close_receptacle,"Done"]
    # object = ["Microwave","Microwave","Sink",["Mug","Sink"],"Microwave",["Mug","Microwave"],"Microwave",None]

    # done = False
    # step = 0
    # while not llm_plan[step] == "Done":
    #     print(llm_plan[step])
    #     print(object[step])
    #     scene_graph_orig, action_feedback = llm_plan[step](scene_graph_orig,controller,object[step],step,tagging_module)
            
    #     step = step+1
    #     with open(scene_graphs_path+"/sg1"+str(step)+".json", "w") as json_file:
    #         json.dump(scene_graph_orig, json_file, indent=2)



    # print("222222222222222222222222222222222222222222222222222222222222222")
    # print("222222222222222222222222222222222222222222222222222222222222222")
    # print("222222222222222222222222222222222222222222222222222222222222222")

    # llm_plan = [navigate,open_receptacle,navigate,pick_object,navigate,put_object,close_receptacle,"Done"]
    # object = ["Fridge","Fridge","Mug","Mug","Fridge",["Mug","Fridge"],"Fridge",None]

    # done = False
    # step = 0
    # while not llm_plan[step] == "Done":
    #     print(llm_plan[step])
    #     print(object[step])
    #     scene_graph, action_feedback = llm_plan[step](scene_graph,controller,object[step],step,tagging_module)
    #     if action_feedback == False:
    #         #Run LLM
    #         #General Case: input: task, previous plan, point where plan failed, new plan
    #         #Specific to pick_place: possible location of receptacle
    #         llm_plan = [navigate,open_receptacle,navigate,open_receptacle,pick_object,navigate,put_object,close_receptacle,"Done"]
    #         object = ["Fridge","Fridge","Microwave","Microwave","Mug","Fridge",["Mug","Fridge"],"Fridge",None]
    #         step = step -1
            
    #     step = step+1
    #     with open(scene_graphs_path+"/sg2"+str(step)+".json", "w") as json_file:
    #         json.dump(scene_graph, json_file, indent=2)




    # print("#########################################################")
    # print("Task Completed")
    # print("##########################################################")



















# #TODO: check https://ai2thor.allenai.org/manipulathor/documentation/#interaction
# #TODO: Gaussain Splatting for memory update
# #TODO: https://openreview.net/forum?id=eJhc_CPXQIT












#   "CounterTop|-01.49|+00.95|+01.32": {
#     "state": "clear",
#     "pose": [
#       [
#         -1.9029698371887207,
#         0.9009991884231567,
#         1.3097600936889648
#       ],
#       [
#         0.0,
#         269.91937255859375,
#         0.0
#       ]
#     ],
#     "camera_pose": 30,
#     "contains": ["Microwave|-02.58|+00.90|+02.44","Plate|-02.35|+00.90|+00.05",
#       "Potato|-02.24|+00.94|-00.18","Spatula|-02.31|+00.91|+00.33","Toaster|-02.57|+00.90|+01.88"]
#   },





















            
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


