from ai2thor_utils import save_frame, get_obj_id
from ai2thor_utils import rotate_angle, get_angle_and_closest_position
from utils import find_closest_items
from sg_utils import update_scene_graph
import os
import pdb


def open_receptacle(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    action_feedback = None
    obj_id = get_obj_id(object, scene_graph)


    action = "Open"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)


    return scene_graph, action_feedback, visual_feedback



def close_receptacle(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    action_feedback = None
    obj_id = get_obj_id(object, scene_graph)

  
    action = "Close"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)


    return scene_graph, action_feedback, visual_feedback



def pick_object(scene_graph,controller,object,action_no,tagging_module,objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    action_feedback = None
    obj_id = get_obj_id(object[0], scene_graph)
    
    recept_id = get_obj_id(object[1], scene_graph)


    action = "Pickup"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)


    return scene_graph, action_feedback, visual_feedback




def put_object(scene_graph,controller,object,action_no,tagging_module,objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    action_feedback = None
    obj_id = get_obj_id(object[0], scene_graph)
    recept_id = get_obj_id(object[1], scene_graph)


    action = "Putdown"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    


    return scene_graph, action_feedback, visual_feedback


def navigate(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    action_feedback = None
    obj_id = get_obj_id(object, scene_graph)        


    return scene_graph, action_feedback, visual_feedback







def crouch(controller):
    controller.step(action="Crouch")

    return None

def stand(controller):
    controller.step(action="Stand")
    # bgr_frame = save_frame(controller,str(action_no),folder_name)

    return None

def camera_rotate(controller,angle=30):
    if angle>0:
        controller.step(
        action="LookUp",
        degrees=angle
            )
    elif angle<0:
        controller.step(
        action="LookDown",
        degrees=-angle
            )
    # bgr_frame = save_frame(controller,str(action_no),folder_name)

    return None

def crouch_n_image(controller,name,folder_name = "action_images/"):
        camera_rotate(controller,30)
        crouch(controller)
        save_frame(controller,str(name),folder_name)
        stand(controller)
        camera_rotate(controller,-30)