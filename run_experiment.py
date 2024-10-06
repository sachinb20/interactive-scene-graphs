import json
from pathlib import Path
import os
from tqdm import trange
import argparse
import pdb
import logging

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

# from lang_sam import LangSAM
from Image_tagging import RAM, Detic
from LLM.GPT_planner import PlanningAgent
from LLM.GPT_replanner import ReplanningAgent

from sg_utils import planner_input
from ai2thor_utils import disable_objects, get_only_obj_ids, metadata_scene_graph
from llm_utils import get_plan_objects, llm_check, ACTION_MAP


def print_n_log(log_file_path, text):
    print(text)
    with open(log_file_path, 'a') as log_file:  # Open in append mode
        log_file.write(f"{text}\n")



# Helper function to set up the experiment based on type
def setup_detector(experiment_type, scene_graph):
    """Configure detect_objects and tagging_module based on experiment type."""
    experiment_config = {
        "obj_introduced": {
            "detect_objects": list(get_only_obj_ids(scene_graph)),
            "tagging_module": Detic(confidence_threshold=0.3),
        },
        "multi_step": {
            "detect_objects": None,
            "tagging_module": None,
        },
        "obj_displaced": {
            "detect_objects": None,
            "tagging_module": Detic(confidence_threshold=0.3),
        },
    }
    return experiment_config.get(experiment_type, {"detect_objects": None, "tagging_module": None}).values()


# Function to handle each experiment type and run the LLM plan
def get_LLM_plan(experiment_type, agent, scene_graph, prompt, details, index, log_file_path):
    if experiment_type == "multi_step":
        return run_multi_step(agent, scene_graph, prompt, log_file_path)

    if experiment_type == "obj_displaced":
        return run_obj_displaced(details, index, agent, scene_graph, prompt, log_file_path)

    if experiment_type == "obj_introduced":
        return run_obj_introduced(details, agent, scene_graph, prompt, log_file_path)

# Handle multi-step experiment type
def run_multi_step(agent, scene_graph, prompt, log_file_path):

    detect_objects, tagging_module = setup_detector("multi_step", scene_graph)
    action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
    for action in action_sequence.actions:
        print_n_log(log_file_path,f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

    llm_plan, obj = get_plan_objects(action_sequence)
    llm_plan, obj = llm_check(llm_plan, obj,scene_graph)
    update_sg = True

    return llm_plan, obj, update_sg, detect_objects, tagging_module

# Handle object displaced experiment type
def run_obj_displaced(details, index, agent, scene_graph, prompt, log_file_path):

    if index < len(details['prompts']) - 1:
        detect_objects, tagging_module = setup_detector("multi_step", scene_graph)
        llm_plan = details['action_seq']["llm_plan"][index]
        obj = details['action_seq']["object"][index]
        llm_plan = [ACTION_MAP[action] for action in llm_plan]
        update_sg = False
    else:
        detect_objects, tagging_module = setup_detector("obj_displaced", scene_graph)
        action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
        for action in action_sequence.actions:
            print_n_log(log_file_path,f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

        llm_plan, obj = get_plan_objects(action_sequence)
        llm_plan, obj = llm_check(llm_plan, obj,scene_graph)
        update_sg = True
    
    return llm_plan, obj, update_sg, detect_objects, tagging_module


# Handle object introduced experiment type
def run_obj_introduced(details, agent, scene_graph, prompt, log_file_path):

    detect_objects, tagging_module = setup_detector("obj_introduced", scene_graph)

    object_removed = details["removed_object"][0]
    scene_graph[object_removed[0]]['contains'].remove(object_removed[1])

    action_sequence = agent.planner(args.embodiment, planner_input(scene_graph), prompt)
    for action in action_sequence.actions:
        print_n_log(log_file_path,f"Action: {action.action}, Object: {action.object}, Target: {action.target}")

    llm_plan, obj = get_plan_objects(action_sequence)
    llm_plan, obj = llm_check(llm_plan, obj,scene_graph)
    update_sg = True

    return llm_plan, obj, update_sg, detect_objects, tagging_module




def main(args: argparse.Namespace):

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
        # platform=CloudRendering 
    )


    disable_objects(controller,scene)



    dataset_root = Path(args.dataset_root).expanduser()
    
    # Paths using dynamic dataset_root, scene, and experiment_type
    
    scene_graph_path = dataset_root / f"sg_data/{args.scene}_sg.json"
    task_file_path = dataset_root / f"tasks/{args.scene}/{args.experiment_type}.json"
    

    # Load the scene graph
    with open(scene_graph_path, 'r') as file:
        scene_graph = json.load(file)
    
    # Open and read the JSON task file
    with open(task_file_path, 'r') as file:
        task_data = json.load(file)






    experiment_type = args.experiment_type
    agent = PlanningAgent()
    replanning_agent = ReplanningAgent()

    # detect_objects, tagging_module = setup_detector(experiment_type, scene_graph)


    for task, details in task_data.items():
        
        log_folder = os.path.join(dataset_root, f"tasks/{args.scene}/{args.experiment_type}")
        os.makedirs(log_folder, exist_ok=True)
        log_file_path = os.path.join(log_folder, f'{task}.txt')

        print_n_log(log_file_path,f"Task: {task}")
        for index, prompt in enumerate(details['prompts']):
            step = 0
            print_n_log(log_file_path,f"Prompt: {prompt}")

            action_images_path = dataset_root / f"action_images/{args.scene}/{args.experiment_type}/{task}/{prompt}"
            llm_plan, object, update_sg, detect_objects, tagging_module = get_LLM_plan(experiment_type, agent, scene_graph, prompt, details, index, log_file_path)
            action_history = None if experiment_type == "obj_introduced" else details['prompts'][:index]

            visual_feedback_count = 0
            action_feedback_count = 0
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(llm_plan)
            print(object)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # Continue until the plan reaches the "standby" step
            while llm_plan[step] != "standby":
                
                # Execute the current step in the LLM plan
                scene_graph, action_feedback, visual_feedback = llm_plan[step](
                    scene_graph, controller, object[step], step, tagging_module, 
                    detect_objects, update_sg, folder_name=action_images_path
                )

                step += 1

                print_n_log(log_file_path,visual_feedback)
                print_n_log(log_file_path,action_feedback)

                # Handle visual feedback if available
                if visual_feedback is not None:
                    print_n_log(log_file_path,json.dumps(planner_input(scene_graph), indent=4))
                    print_n_log(log_file_path,f"Prompt: {prompt}, Action history: {action_history}, Visual feedback: {visual_feedback}, Feedback count: {visual_feedback_count}")
                    
                    # Replan the sequence based on feedback
                    replanned_sequence = replanning_agent.replanner(
                        args.embodiment, planner_input(scene_graph), prompt, 
                        action_history, visual_feedback
                    )
                    
                    # Update feedback count and check feedback limit
                    visual_feedback_count += 1
                    if visual_feedback_count == args.visual_feedback_limit:
                        detect_objects = None
                        tagging_module = None
                    
                    # Log replanned actions and update the LLM plan
                    for action in replanned_sequence.actions:
                        print_n_log(log_file_path,f"Action: {action.action}, Object: {action.object}, Target: {action.target}")
                    
                    llm_plan, object = get_plan_objects(replanned_sequence)
                    llm_plan, object = llm_check(llm_plan, object,scene_graph)
                    
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print(llm_plan)
                    print(object)
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # Reset step counter after replanning
                    step = 0

                # Handle action feedback if available
                if action_feedback is not None:
                    print_n_log(log_file_path,json.dumps(planner_input(scene_graph), indent=4))
                    action_feedback 
                    print_n_log(log_file_path,f"Prompt: {prompt}, Action history: {action_history}, Action feedback: {action_feedback}, Feedback count: {action_feedback_count}")
                    
                    # Replan the sequence based on feedback
                    replanned_sequence = replanning_agent.replanner(
                        args.embodiment, planner_input(scene_graph), prompt, 
                        action_history, action_feedback
                    )
                    
                    # Update feedback count and check feedback limit
                    action_feedback_count += 1
                    if action_feedback_count == args.action_feedback_limit:
                       
                        if scene_graph["Agent"]["contains"] !=[]:
                            print("haath me kuch tha")
                            controller.step(
                            action="DropHandObject",
                            forceAction=True)
                            
                        break
                    
                    # Log replanned actions and update the LLM plan
                    for action in replanned_sequence.actions:
                        print_n_log(log_file_path,f"Action: {action.action}, Object: {action.object}, Target: {action.target}")
                    
                    llm_plan, object = get_plan_objects(replanned_sequence)
                    llm_plan, object = llm_check(llm_plan, object,scene_graph)

                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print(llm_plan)
                    print(object)
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # Reset step counter after replanning
                    step = 0

            print_n_log(log_file_path,json.dumps(planner_input(scene_graph), indent=4))
            print_n_log(log_file_path,"_" * 50)
            # controller.step(
            #     action="OpenObject",
            #     objectId="Fridge|-02.48|+00.00|-00.78",
            #     openness=1,
            #     forceAction=True
            # )
            print_n_log(log_file_path,json.dumps(planner_input(metadata_scene_graph(controller, scene_graph)), indent=4))
            
            
            
        print_n_log(log_file_path,"=" * 50)
        controller.reset()
        
        
    
        






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
        default="obj_introduced",
        choices=["multi_step", "obj_displaced", "obj_introduced"],
        help="The type of experiment, e.g., obj_introduced, etc.",
    )

    parser.add_argument(
        "--scene",
        default="FloorPlan4",
        type=str,
        help="Scene Name"
    )

    parser.add_argument(
        "--visual_feedback_limit",
        default=1,
        type=int,
        help="Number of times feedback is taken from the scene"
    )

    parser.add_argument(
        "--action_feedback_limit",
        default=2,
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


