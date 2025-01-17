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

from Image_tagging import RAM, Detic
from LLM.GPT_planner import PlanningAgent
from LLM.GPT_replanner import ReplanningAgent
import math
from typing import List,Dict
import json
# import open_clip
import torch
import torchvision

import copy
from sg_utils import planner_input
from ai2thor_utils import disable_objects, get_only_obj_ids

from llm_utils import get_plan_objects, llm_check, ACTION_MAP
import pdb



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
    scene = args.scene
    # Initialize the controller
    controller = Controller(
        agentMode=args.agent_mode,
        visibilityDistance=args.visibility_distance,
        scene=args.scene,
        gridSize=args.grid_size,
        snapToGrid=args.snap_to_grid,
        rotateStepDegrees=args.rotate_step_degrees,
        renderDepthImage=args.render_depth_image,
        renderInstanceSegmentation=args.render_instance_segmentation,
        renderSemanticSegmentation=args.render_semantic_segmentation,
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,  # Assuming this is predefined elsewhere
    )


    disable_objects(controller,scene)



    dataset_root = Path(args.dataset_root).expanduser()
    
    # Paths using dynamic dataset_root, scene, and experiment_type
    action_images_path = dataset_root / f"action_images/{args.scene}/{args.experiment_type}/"
    scene_graph_path = dataset_root / f"sg_data/{args.scene}_sg.json"
    task_file_path = dataset_root / f"tasks/{args.scene}/{args.experiment_type}.json"

    # Load the scene graph
    with open(scene_graph_path, 'r') as file:
        scene_graph = json.load(file)
    
    # Open and read the JSON task file
    with open(task_file_path, 'r') as file:
        json_data = json.load(file)


    experiment_type = args.experiment_type


    if experiment_type == "obj_introduced":

        detect_objects = list(get_only_obj_ids(scene_graph))

        tagging_module = Detic()

    elif experiment_type == "multi_step":

        detect_objects = None

        tagging_module = None

    elif experiment_type == "obj_displaced":

        detect_objects = None

        tagging_module = Detic()


    agent = PlanningAgent()
    replanning_agent = ReplanningAgent()









    # Iterate through the JSON data and print each prompt
    for task, details in json_data.items():
        print(f"Task: {task}")
        
        for index, prompt in enumerate(details['prompts']):
            step = 0
            print(f"Prompt: {prompt}")

            


            #Run LLM 
            if experiment_type == "multi_step":     
                
                action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
                # Print the structured output
                for action in action_sequence.actions:
                    print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

                llm_plan,object = get_plan_objects(action_sequence)
                llm_plan,object = llm_check(llm_plan,object)
                update_sg = True


            if experiment_type == "obj_displaced":
                if index<len(details['prompts'])-1:
                    llm_plan = details['action_seq']["llm_plan"][index]
                    object = details['action_seq']["object"][index]

                    print(llm_plan)
                    
                    llm_plan = [ACTION_MAP[action] for action in llm_plan]
                    update_sg = False

                else:
                    
                    action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
                    # Print the structured output
                    for action in action_sequence.actions:
                        print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

                    llm_plan,object = get_plan_objects(action_sequence)

                    llm_plan,object = llm_check(llm_plan,object)
                    action_history = details['prompts'][:index]
                    update_sg = False
            

            if experiment_type == "obj_introduced":     
                object_removed = details["removed_object"][0]
                # controller.step(action="DisableObject",objectId=object_removed)
                scene_graph[object_removed[0]]['contains'].remove(object_removed[1])

                action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
                # Print the structured output
                for action in action_sequence.actions:
                    print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

                llm_plan,object = get_plan_objects(action_sequence)
                llm_plan,object = llm_check(llm_plan,object)
                action_history = None
                update_sg = True





            feedback_count = 0

            # Continue until the plan reaches the "standby" step
            while llm_plan[step] != "standby":
                print(llm_plan[step])
                print(object[step])

                # Execute the current step in the LLM plan
                scene_graph, action_feedback, visual_feedback = llm_plan[step](
                    scene_graph, controller, object[step], step, tagging_module, 
                    detect_objects, update_sg, folder_name=action_images_path
                )
                
                step += 1
                print(visual_feedback)

                # Handle visual feedback if available
                if visual_feedback is not None:
                    print(planner_input(scene_graph), prompt, action_history, visual_feedback, feedback_count)
                    
                    # Replan the sequence based on feedback
                    replanned_sequence = replanning_agent.replanner(
                        args.embodiment, planner_input(scene_graph), prompt, 
                        action_history, visual_feedback
                    )
                    
                    # Update feedback count and check feedback limit
                    feedback_count += 1
                    if feedback_count == args.feedback_limit:
                        detect_objects = None
                        tagging_module = None
                    
                    # Log replanned actions and update the LLM plan
                    for action in replanned_sequence.actions:
                        print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")
                    
                    llm_plan, object = get_plan_objects(replanned_sequence)
                    llm_plan, object = llm_check(llm_plan, object)
                    
                    # Reset step counter after replanning
                    step = 0


        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        break










# #TODO: check https://ai2thor.allenai.org/manipulathor/documentation/#interaction
# #TODO: Gaussain Splatting for memory update
# #TODO: https://openreview.net/forum?id=eJhc_CPXQIT


































def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Program Arguments")
    
    # Existing arguments
    # Updated dataset root path
    parser.add_argument(
        "--dataset_root",
        default=str(Path("/home/hypatia/Sachin_Workspace/interactive-scene-graphs/").expanduser()),
        help="The root path to the dataset."
    )
    
    parser.add_argument(
        "--experiment_type",
        default="multi_step",
        choices=["multi_step", "obj_displaced", "obj_introduced"],
        help="The type of experiment, e.g., obj_introduced, etc.",
    )

    parser.add_argument(
        "--scene",
        default="FloorPlan6",
        type=str,
        help="Scene Name"
    )

    parser.add_argument(
        "--feedback_limit",
        default=1,
        type=int,
        help="Number of times feedback is taken from the scene"
    )



    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--height", default=480, type=int, help="The height of the image."
    )
    parser.add_argument(
        "--width", default=320, type=int, help="The width of the image."
    )
    parser.add_argument(
        "--fov", default=90, type=int, help="The (vertical) field of view of the camera."
    )

    parser.add_argument(
    "--embodiment",
    default="Bi-manipulation",
    type=str,
    help="The embodiment type, e.g., Bi-manipulation, Single-arm, etc."
    )

    # New arguments for controller
    parser.add_argument(
        "--agent_mode",
        default="default",
        type=str,
        choices=["default", "arm"],
        help="The mode of the agent. Choose between 'default' and 'arm'."
    )

    parser.add_argument(
        "--rotate_step_degrees",
        default=30,
        type=int,
        help="The rotation step in degrees."
    )
    
    parser.add_argument(
        "--visibility_distance",
        default=1.5,
        type=float,
        help="The visibility distance of the agent."
    )
    
    parser.add_argument(
        "--grid_size",
        default=0.1,
        type=float,
        help="The size of the grid for movement."
    )
    
    parser.add_argument(
        "--snap_to_grid",
        default=False,
        type=bool,
        help="Whether the agent's movement snaps to the grid."
    )
    

    
    # Image modality options
    parser.add_argument(
        "--render_depth_image",
        default=False,
        type=bool,
        help="Whether to render the depth image."
    )
    
    parser.add_argument(
        "--render_instance_segmentation",
        default=False,
        type=bool,
        help="Whether to render instance segmentation."
    )
    
    parser.add_argument(
        "--render_semantic_segmentation",
        default=False,
        type=bool,
        help="Whether to render semantic segmentation."
    )
    
    return parser




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args)


