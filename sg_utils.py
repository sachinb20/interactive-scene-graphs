import json
from collections import defaultdict
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import os
import copy

# def reverse_json(data):

    
#     # Create a dictionary to hold the reversed data
#     reversed_data = defaultdict(lambda: {"parentReceptacles": [], "ObjectState": None})
    
#     # Iterate through the original data
#     for obj, details in data.items():
#         # Handle objects with parent receptacles
#         if details["parentReceptacles"]:
#             for parent in details["parentReceptacles"]:
#                 reversed_data[parent]["parentReceptacles"].append(obj)
#         # Handle objects with ObjectState
#         if details["ObjectState"] is not None:
#             reversed_data[obj]["ObjectState"] = details["ObjectState"]
    
#     # Create the final scene list
#     scene = []
#     print(reversed_data)
#     for parent, details in reversed_data.items():
#         # Determine the state of the parent object
#         parent_state = details["ObjectState"] if details["ObjectState"] else "clear"
        
#         # Create the scene entry for the parent object
#         scene_entry = {
#             "object": parent,
#             "state": parent_state,
#             "contains": []
#         }
        
#         for child in details["parentReceptacles"]:
#             # Determine the state of the contained object
#             child_state = "in" if parent_state == "Closed" else "on"
            
#             # Create the entry for the contained object
#             child_entry = {
#                 "object": child,
#                 "state": child_state
#             }
#             scene_entry["contains"].append(child_entry)
        
#         scene.append(scene_entry)
    
#     # Prepare the final output dictionary
#     final_output = {"scene": scene}

#     return final_output


def planner_input(scene_graph):
  sg = {}
  for key, value in scene_graph.items():
      sg[key] = {
          "contains": value["contains"],
          "State": value["state"]}
      
  return sg



def reverse_json(data):

    
    # Create a dictionary to hold the reversed data
    reversed_data = defaultdict(lambda: {"parentReceptacles": [], "ObjectState": None})
    
    # Iterate through the original data
    for obj, details in data.items():
        # Handle objects with parent receptacles
        if details["parentReceptacles"]:
            for parent in details["parentReceptacles"]:
                reversed_data[parent]["parentReceptacles"].append(obj)
        # Handle objects with ObjectState
        if details["ObjectState"] is not None:
            reversed_data[obj]["ObjectState"] = details["ObjectState"]
    
    # Create the final scene list
    scene = {}
    print(reversed_data)
    for parent, details in reversed_data.items():
        # Determine the state of the parent object
        parent_state = details["ObjectState"] if details["ObjectState"] else "clear"
        
        # Create the scene entry for the parent object
        scene[parent] = {
            "state": parent_state,
            "pose": [[]],
            "camera_pose": 30,
            "crouch": "False",
            "contains": []
        }
        
        for child in details["parentReceptacles"]:

            scene[parent]["contains"].append(child)
            
    # Prepare the final output dictionary
    final_output = {"scene": scene}

    return final_output


def filter_sg(data):
    # Extract desired keys and values
    result = {}
    for key, value in data.items():
        result[key] = {
            "parentReceptacles": value["parentReceptacles"],
            "ObjectState": value["ObjectState"]
        }

    return result


def create_scene_graph_from_metadata(objects):
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
    
    scene_graph["Stove"]={  
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
            "parentReceptacles": None,
            "ObjectState": None
            }
    

    OBJECT_LIST = []
    for obj in objects:
        obj_id = obj["objectId"]
        aabb = obj["objectOrientedBoundingBox"]["cornerPoints"] if obj["pickupable"] else obj["axisAlignedBoundingBox"]["cornerPoints"]
        if "Stove" in obj_id:
            obj["parentReceptacles"] = ["Stove"]
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

# def update_scene_graph(scene_graph,action,obj_id,recept_id):

#     if action == "Pickup":
#         scene_graph[obj_id]['parentReceptacles'] = ["Agent"]

#     elif action == "Putdown":
#         scene_graph[obj_id]['parentReceptacles'] = [recept_id]

#     elif action == "Open":
#         scene_graph[obj_id]['ObjectState'] = "Open"

#     elif action == "Close":
#         scene_graph[obj_id]['ObjectState'] = "Close"

#     elif action == "Navigate":
#         scene_graph = scene_graph

#     return scene_graph

def update_scene_graph(scene_graph, action, obj_id, recept_id=None):
    try:
        if action == "Pickup":
            try:
                scene_graph["Agent"]['contains'].append(obj_id)
                scene_graph[recept_id]['contains'].remove(obj_id)
            except KeyError as e:
                print(f"KeyError during Pickup: {e}")
            except ValueError as e:
                print(f"ValueError during Pickup: {e}")

        elif action == "Putdown":
            try:
                scene_graph[recept_id]['contains'].append(obj_id)
                scene_graph["Agent"]['contains'].remove(obj_id)
            except KeyError as e:
                print(f"KeyError during Putdown: {e}")
            except ValueError as e:
                print(f"ValueError during Putdown: {e}")

        elif action == "Open":
            try:
                scene_graph[obj_id]['state'] = "Open"
            except KeyError as e:
                print(f"KeyError during Open: {e}")

        elif action == "Close":
            try:
                scene_graph[obj_id]['state'] = "Closed"
            except KeyError as e:
                print(f"KeyError during Close: {e}")

        elif action == "Navigate":
            try:
                # You may want to implement actual navigation logic here
                scene_graph = scene_graph
            except Exception as e:
                print(f"Error during Navigate: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return scene_graph



def create_sg():
    
    # for i in range(30):

      scene = "FloorPlan_Val3_4"
      # Initialize the controller
      controller = Controller(
          # agentMode = "arm",
          agentMode="default",
          visibilityDistance=1.5,
          # scene=get_scene(args.scene_name),
          scene=scene,
          # step sizes
          gridSize=0.5,
          snapToGrid=False,
          rotateStepDegrees=30,
          # image modalities
          renderDepthImage=True,
          renderInstanceSegmentation=True,
          renderSemanticSegmentation=True,
          # camera properties
          width=640,
          height=480,
          fieldOfView=90,
          platform=CloudRendering,
      )



      path = "/home/hypatia/Sachin_Workspace/interactive-scene-graphs"
      action_images = "action_images"
      scene_graphs = "sg_data/"
      originals = "originals/"

      action_images_path = os.path.join(path,action_images)
      scene_graphs_path =  os.path.join(path,scene_graphs)
      scene_graphs_og_path = os.path.join(scene_graphs_path,originals)

      if not os.path.exists(action_images_path):
          os.mkdir(action_images_path)

      if not os.path.exists(scene_graphs_path):
          os.mkdir(scene_graphs_path)

      if not os.path.exists(scene_graphs_path):
          os.mkdir(scene_graphs_og_path)


      scene_graph, object_list = create_scene_graph_from_metadata(controller.last_event.metadata["objects"])
      scene_graph_orig = scene_graph

      scene_graph_orig = copy.deepcopy(scene_graph)       

      scene_graph = filter_sg(scene_graph)
      # Save scene graph to a file
      # file_path = scene_graphs_path+scene +"_filter_sg.json"
      # with open(file_path, "w") as json_file:
      #     json.dump(scene_graph, json_file, indent=2)

      # print(f"Filtered Scene graph saved to {file_path}")

      scene_graph = reverse_json(scene_graph)
      # Save scene graph to a file
      file_path = scene_graphs_path+scene +"_sg.json"
      with open(file_path, "w") as json_file:
          json.dump(scene_graph, json_file, indent=2)

      print(f"Reversed Scene graph saved to {file_path}")

      file_path = scene_graphs_og_path+scene +"_original_sg.json"
      with open(file_path, "w") as json_file:
          json.dump(scene_graph, json_file, indent=2)

      


# create_sg()

