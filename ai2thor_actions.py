from ai2thor_utils import save_frame, get_obj_id
from ai2thor_utils import rotate_angle, get_angle_and_closest_position
from utils import find_closest_items
from isg import create_scene_graph, update_scene_graph




def open_receptacle(scene_graph,controller,object,action_no,tagging_module):

    obj_id = get_obj_id(object, scene_graph)
    #check if object already open
    state = 'close'

    if state == 'close':
        controller.step(
            action="OpenObject",
            objectId=obj_id,
            openness=1,
            forceAction=False
        )

        save_frame(controller,str(action_no))

        #Verify Action 
        action_feedback = True

    # #Update SG
    action = "Open"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph, action_feedback

def close_receptacle(scene_graph,controller,object,action_no,tagging_module):

    obj_id = get_obj_id(object, scene_graph)
    #Check if object already closed
    state = 'open'

    if state == 'open':
        controller.step(
        action="CloseObject",
        objectId=obj_id,
        forceAction=False
        )
        
        save_frame(controller,str(action_no))
        
        #Verify Action 
        action_feedback = True

    # #Update SG
    action = "Close"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph, action_feedback

def pick_object(scene_graph,controller,object,action_no,tagging_module):

    obj_id = get_obj_id(object, scene_graph)
    event = controller.step(
    action="PickupObject",
    objectId=obj_id,
    forceAction=False,
    manualInteract=False
    )

    bgr_frame = save_frame(controller,str(action_no))

    #Verify Action
    action_feedback = True

    #Update SG
    action = "Pickup"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,None)

    return scene_graph, action_feedback

def put_object(scene_graph,controller,object,action_no,tagging_module):

    obj_id = get_obj_id(object[0], scene_graph)
    recept_id = get_obj_id(object[1], scene_graph)
    controller.step(
    action="PutObject",
    objectId=recept_id,
    forceAction=False,
    placeStationary=True
    )

    save_frame(controller,str(action_no))


    #Verify
    action_feedback = True


    action = "Putdown"
    scene_graph = update_scene_graph(scene_graph,action,obj_id,recept_id)    

    
    # obj_id = get_obj_id(object, scene_graph)
    # controller.step(
    # action="PutObject",
    # objectId=obj_id,
    # forceAction=False,
    # placeStationary=True
    # )

    # save_frame(controller,str(action_no))


    # #Verify
    # action = "Putdown"
    # scene_graph = update_scene_graph(scene_graph,action,obj_id,obj_id)    

    return scene_graph, action_feedback



def navigate(scene_graph,controller,object,action_no,tagging_module):

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


    bgr_frame = save_frame(controller,str(action_no))

    #Navigation Verification
    closest_items = find_closest_items(controller.last_event.metadata["agent"]["position"], scene_graph, num_items=10)
    caption, text_prompt = tagging_module.predict(bgr_frame)
    print(text_prompt)
    print(closest_items)

    return scene_graph, True