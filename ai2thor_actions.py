from ai2thor_utils import save_frame, get_obj_id
from ai2thor_utils import rotate_angle, get_angle_and_closest_position
from utils import find_closest_items
from sg_utils import update_scene_graph
import os
import pdb


def open_receptacle(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):
    visual_feedback = None
    obj_id = get_obj_id(object, scene_graph)

    #crouch if required
    if scene_graph[obj_id]["crouch"] =="True":
        camera_rotate(controller,30)
        crouch(controller)


    state_image = None
    #check if object already open

    state = 'close'

    if state == 'close':
        controller.step(
            action="OpenObject",
            objectId=obj_id,
            openness=1,
            forceAction=False
        )
        action_feedback = f"Failed to open {obj_id}" if not controller.last_event.metadata["lastActionSuccess"] else None

        save_frame(controller,str(action_no),folder_name)

        #Verify Action 

        


    #Update SG
    if update_sg and action_feedback is None:
        action = "Open"
        scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    if scene_graph[obj_id]["state"] == "Open" and tagging_module is not None and detect_objects is not None:
        
        image_path = os.path.join(folder_name, str(action_no)+'.jpg')
        output_path = os.path.join(folder_name, f"{object}_open_detections.jpg")
        object_predictions, vis_output = tagging_module.tag("custom",",".join([item.split("|")[0] for item in detect_objects]) , image_path) 

        predictions = [detect_objects[i] for i in object_predictions['instances'].pred_classes.tolist()]
        vis_output.save(output_path)  

        #check objects in hand of agent and on/in receptacle
        diff_1_to_2 = list(set(scene_graph[obj_id]["contains"]+scene_graph["Agent"]["contains"]) - set(predictions))

        # Find elements in list2 but not in list1
        diff_2_to_1 = list(set(predictions) - set(scene_graph[obj_id]["contains"]+scene_graph["Agent"]["contains"]))

        # Combined differences
        differences = diff_1_to_2 + diff_2_to_1
        new_objects = ",".join([item.split("|")[0] for item in diff_2_to_1])
        visual_feedback = None if diff_2_to_1 == [] else f"New objects {new_objects} have been detected in the {object}, replan the sequence."

        if visual_feedback is not None:
            return scene_graph, None, visual_feedback


    #uncrouch if required
    if scene_graph[obj_id]["crouch"] =="True":
        camera_rotate(controller,-30)
        stand(controller)

    return scene_graph, action_feedback, visual_feedback

















def close_receptacle(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):

    obj_id = get_obj_id(object, scene_graph)

    #check if crouch
    if scene_graph[obj_id]["crouch"] =="True":
        camera_rotate(controller,30)
        crouch(controller)

    #Check if object already closed
    state = 'open'

    # if scene_graph[obj_id]["state"] == "Open" and tagging_module is not None and detect_objects is not None:
    #     save_frame(controller,"temp",folder_name)
    #     image_path = os.path.join(folder_name, "temp"+'.jpg')
    #     output_path = os.path.join(folder_name, f"{object}_close_detections.jpg")
    #     object_predictions, vis_output = tagging_module.tag("custom", detect_objects, image_path)         
    #     vis_output.save(output_path)  

    visual_feedback = None

    if state == 'open':
        controller.step(
        action="CloseObject",
        objectId=obj_id,
        forceAction=False
        )
        action_feedback = f"Failed to close {obj_id}" if not controller.last_event.metadata["lastActionSuccess"] else None

        save_frame(controller,str(action_no),folder_name)
        
        #Verify Action 

    # #Update SG
    if update_sg and action_feedback is None:
        action = "Close"
        scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    if scene_graph[obj_id]["crouch"] =="True":
        camera_rotate(controller,-30)
        stand(controller)

    return scene_graph, action_feedback, visual_feedback

















def pick_object(scene_graph,controller,object,action_no,tagging_module,objects,update_sg, folder_name = "action_images/"):

    obj_id = get_obj_id(object[0], scene_graph)
    
    recept_id = get_obj_id(object[1], scene_graph)

    visual_feedback = None
    if tagging_module is not None:
        save_frame(controller,"temp",folder_name)
        image_path = os.path.join(folder_name, "temp"+'.jpg')
        output_path = os.path.join(folder_name, f"{obj_id}.jpg")
        object_predictions, vis_output = tagging_module.tag("custom",object[0].split("|")[0], image_path)
        # pdb.set_trace()
        op = object_predictions['instances'].pred_classes.tolist()
        vis_output.save(output_path)  
        visual_feedback = None if len(op)>0 else f"The {obj_id} is not on/in the {recept_id}. Find another place to pick the {object}"
        if visual_feedback is not None:
            return scene_graph, None, visual_feedback

    

    controller.step(
    action="PickupObject",
    objectId=obj_id,
    forceAction=False,
    manualInteract=False
    )

    action_feedback = f"Failed to pick {obj_id} in {recept_id}. Skip this action and move to the rest of plan." if not controller.last_event.metadata["lastActionSuccess"] else None

    if scene_graph[recept_id]["crouch"] =="True":
        camera_rotate(controller,30)
        stand(controller)

    bgr_frame = save_frame(controller,str(action_no),folder_name)
    #Verify Action

    #Update SG
    if update_sg and action_feedback is None:
        action = "Pickup"
        scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)

    if scene_graph[recept_id]["crouch"] =="True":
        camera_rotate(controller,-30)
        stand(controller)

    return scene_graph, action_feedback, visual_feedback




def put_object(scene_graph,controller,object,action_no,tagging_module,objects,update_sg, folder_name = "action_images/"):
    action_feedback = None
    obj_id = get_obj_id(object[0], scene_graph)
    recept_id = get_obj_id(object[1], scene_graph)



    controller.step(
    action="PutObject",
    objectId=recept_id,
    forceAction=False,
    placeStationary=True
    )

    if update_sg and scene_graph["Agent"]["contains"][0] == obj_id:
        action_feedback = f"Failed to place {obj_id} in {recept_id}. {obj_id} is in the hand of the Agent, Place it on some other receptacle and continue with the plan." if not controller.last_event.metadata["lastActionSuccess"] else None
    
    elif update_sg and scene_graph["Agent"]["contains"][0] != obj_id:
        action_feedback = f"Failed to place {obj_id} in {recept_id}. {obj_id} is in not in the hand of the Agent, Find the {obj_id} first." if not controller.last_event.metadata["lastActionSuccess"] else None


    if scene_graph[recept_id]["crouch"] =="True":
        camera_rotate(controller,30)
        stand(controller)

    save_frame(controller,str(action_no),folder_name)


    #Verify

    if update_sg and action_feedback is None:
        action = "Putdown"
        scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    

    visual_feedback = None
    # obj_id = get_obj_id(object, scene_graph)
    # controller.step(
    # action="PutObject",
    # objectId=obj_id,
    # forceAction=False,
    # placeStationary=True
    # )

    # save_frame(controller,str(action_no),folder_name)


    # #Verify
    # action = "Putdown"
    # scene_graph = update_scene_graph(scene_graph,action,obj_id,obj_id)    

    if scene_graph[recept_id]["crouch"] =="True":
        camera_rotate(controller,-30)
        stand(controller)

    return scene_graph, action_feedback, visual_feedback























def navigate(scene_graph,controller,object,action_no,tagging_module,detect_objects,update_sg, folder_name = "action_images/"):

    find_suitable_pos_rot = False
    if find_suitable_pos_rot:
        angle_deg, closest, obj_id = get_angle_and_closest_position(controller,object,scene_graph)
        controller.step(action="Teleport", **closest)
        controller.step(
        action="RotateRight",
        degrees=60
        )
        
        angle = rotate_angle(controller, object, scene_graph)
        if angle == None:
            print("replan kr bhidu")

            return scene_graph, False

        controller.step(
            action="RotateRight",
            degrees=angle
        )
        #Navigation Verification
        closest_items = find_closest_items(controller.last_event.metadata["agent"]["position"], scene_graph, num_items=10)
        caption, text_prompt = tagging_module.predict(bgr_frame)
        print(text_prompt)
        print(closest_items)



    elif not find_suitable_pos_rot: 
        obj_id = get_obj_id(object, scene_graph)        

        try:
            posn = {
                'x': scene_graph[obj_id]["pose"][0][0],
                'y': scene_graph[obj_id]["pose"][0][1],
                'z': scene_graph[obj_id]["pose"][0][2]
            }
            rotn = {
                'x': scene_graph[obj_id]["pose"][1][0],
                'y': scene_graph[obj_id]["pose"][1][1],
                'z': scene_graph[obj_id]["pose"][1][2]
            }
            camera_pose = scene_graph[obj_id]["camera_pose"]

        
        except (KeyError, IndexError) as e:
            # Handle missing keys or out of range indexes
            print(f"Error: {str(e)}. Check the structure of the scene graph or the obj_id.")
            return scene_graph, f"Failed to navigate to {obj_id}", None
        
        controller.step(
        action="Teleport",
        position=posn,
        rotation=rotn,
        horizon=camera_pose,
        standing=True
        )

        action_feedback = f"Failed to navigate to {obj_id}" if not controller.last_event.metadata["lastActionSuccess"] else None

    visual_feedback = None
    bgr_frame = save_frame(controller,str(action_no),folder_name)

    if scene_graph[obj_id]["state"] == "clear" and tagging_module is not None and detect_objects is not None:
        
        image_path = os.path.join(folder_name, str(action_no)+'.jpg')
        output_path = os.path.join(folder_name, f"{object}detections.jpg")
        object_predictions, vis_output = tagging_module.tag("custom",",".join([item.split("|")[0] for item in detect_objects]) , image_path) 

        predictions = [detect_objects[i] for i in object_predictions['instances'].pred_classes.tolist()]
        vis_output.save(output_path)  

        #check objects in hand of agent and on/in receptacle
        diff_1_to_2 = list(set(scene_graph[obj_id]["contains"]+scene_graph["Agent"]["contains"]) - set(predictions))

        # Find elements in list2 but not in list1
        diff_2_to_1 = list(set(predictions) - set(scene_graph[obj_id]["contains"]+scene_graph["Agent"]["contains"]))

        # Combined differences
        differences = diff_1_to_2 + diff_2_to_1
        # print(diff_2_to_1)
        scene_graph[object]['contains'] = scene_graph[object]['contains'] + diff_2_to_1

        new_objects = ",".join([item.split("|")[0] for item in diff_2_to_1])
        visual_feedback = None if diff_2_to_1 == [] else f"New objects {new_objects} have been detected on the {object}, replan the sequence."
        
        if visual_feedback is not None:
            return scene_graph, None, visual_feedback

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