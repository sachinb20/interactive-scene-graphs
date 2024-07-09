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
from Image_tagging import RAM,LLAVA


import math
from typing import List,Dict
import json
# import open_clip
import torch
import torchvision

import copy
from isg import create_scene_graph, update_scene_graph
from ai2thor_actions import open_receptacle, close_receptacle, pick_object, put_object,navigate 

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
    scene_graph_orig = scene_graph

    scene_graph_orig = copy.deepcopy(scene_graph)       

    # Save scene graph to a file
    file_path = "scene_graph.json"
    with open(file_path, "w") as json_file:
        json.dump(scene_graph, json_file, indent=2)

    print(f"Scene graph saved to {file_path}")

    tagging_module = RAM()
    # query = "Describe the central object in the image."
    # image_file = f"action_images/1.jpg"
    # op = tagging_module.tag(query,image_file)
    # print(op)
    # updated_obj_id = change_scene(controller)
    # print(updated_obj_id)




    #Language Query to Decide Planning (LLM Planner) 
    #Eg: Pick the Tomato from the sink and place it in the Fridge

    #TODO: check https://ai2thor.allenai.org/manipulathor/documentation/#interaction
    #TODO: Gaussain Splatting for memory update
    #TODO: https://openreview.net/forum?id=eJhc_CPXQIT

    # prompt = "I'd like to have my apple chilled. Could you find a cool place to keep it."
    # agent = PickAndPlaceAgent()
    # planner = agent.pick_and_place(object_list,prompt)




    # print(planner)
    # target_receptacle = planner["target_receptacle"]
    # object = planner["source_object"]

    # target_receptacle = "Fridge"
    # object = "Apple"

    # source_receptacle = scene_graph[get_obj_id(object, scene_graph)]["parentReceptacles"]

    # print(target_receptacle)
    # print(source_receptacle)
    # print(object)


    #Run LLM


    llm_plan = [navigate,open_receptacle,navigate,pick_object,navigate,put_object,close_receptacle,"Done"]
    object = ["Microwave","Microwave","Mug","Mug","Microwave",["Mug","Microwave"],"Microwave",None]

    done = False
    step = 0
    while not llm_plan[step] == "Done":
        print(llm_plan[step])
        print(object[step])
        scene_graph_orig, action_feedback = llm_plan[step](scene_graph_orig,controller,object[step],step,tagging_module)
            
        step = step+1
        with open("sg1"+str(step)+".json", "w") as json_file:
            json.dump(scene_graph_orig, json_file, indent=2)



    print("222222222222222222222222222222222222222222222222222222222222222")
    print("222222222222222222222222222222222222222222222222222222222222222")
    print("222222222222222222222222222222222222222222222222222222222222222")

    llm_plan = [navigate,open_receptacle,navigate,pick_object,navigate,put_object,close_receptacle,"Done"]
    object = ["Fridge","Fridge","Mug","Mug","Fridge",["Mug","Fridge"],"Fridge",None]

    done = False
    step = 0
    while not llm_plan[step] == "Done":
        print(llm_plan[step])
        print(object[step])
        scene_graph, action_feedback = llm_plan[step](scene_graph,controller,object[step],step,tagging_module)
        if action_feedback == False:
            #Run LLM
            #General Case: input: task, previous plan, point where plan failed, new plan
            #Specific to pick_place: possible location of receptacle
            llm_plan = [navigate,open_receptacle,navigate,open_receptacle,pick_object,navigate,put_object,close_receptacle,"Done"]
            object = ["Fridge","Fridge","Microwave","Microwave","Mug","Fridge",["Mug","Fridge"],"Fridge",None]
            step = step -1
            
        step = step+1
        with open("sg2"+str(step)+".json", "w") as json_file:
            json.dump(scene_graph, json_file, indent=2)

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
