from ai2thor_actions import open_receptacle, close_receptacle, pick_object, put_object,navigate 

ACTION_MAP = {
    "open_receptacle": open_receptacle,
    "pick_object": pick_object,
    "navigate": navigate,
    "put_object": put_object,
    "close_receptacle": close_receptacle,
    "standby": "standby"
}

def find_container(input_string, scene_graph):
    if input_string in scene_graph:
        return input_string  # If input is a key, return it directly
    
    for key, value in scene_graph.items():
        if "contains" in value and input_string in value["contains"]:
            return key
    
    return None  # Return None if no match is found


def get_plan_objects(action_sequence):
    
    llm_plan = []
    objects = []

    # Action to function mapping


    # Iterate over the actions in the sequence and build the plan
    for action in action_sequence.actions:
        # Append the mapped function to llm_plan
        llm_plan.append(ACTION_MAP.get(action.action))
        
        # Handle objects and targets
        if action.target:
            objects.append([action.object, action.target])
        else:
            objects.append(action.object)


    return llm_plan,objects



def llm_check(actions,objects, scene_graph):
    # To store the last known navigate object
    last_navigate_object = None

    # Iterate over the actions and objects to update "None" in pick_object or put_object
    for i, action in enumerate(actions):
        if action == navigate:
            objects[i] = find_container(objects[i] , scene_graph)
            last_navigate_object = objects[i]
        elif action in [pick_object, put_object]:
            if type(objects[i]) == str:
                # print(333)
                objects[i] = [objects[i],last_navigate_object]

    return actions, objects