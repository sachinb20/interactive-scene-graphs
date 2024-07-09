
def create_scene_graph(objects):
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
    

    OBJECT_LIST = []
    for obj in objects:
        obj_id = obj["objectId"]
        aabb = obj["objectOrientedBoundingBox"]["cornerPoints"] if obj["pickupable"] else obj["axisAlignedBoundingBox"]["cornerPoints"]

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

def update_scene_graph(scene_graph,action,obj_id,recept_id):

    if action == "Pickup":
        scene_graph[obj_id]['parentReceptacles'] = ["Agent"]

    elif action == "Putdown":
        scene_graph[obj_id]['parentReceptacles'] = [recept_id]

    elif action == "Open":
        scene_graph[obj_id]['ObjectState'] = "Open"

    elif action == "Close":
        scene_graph[obj_id]['ObjectState'] = "Close"

    elif action == "Navigate":
        scene_graph = scene_graph

    return scene_graph