from typing import List,Dict
import math


def process_tag_classes(text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]):
    '''
    Convert a text prompt from Tag2Text to a list of classes. 
    '''
    classes = text_prompt.split(',')
    classes = [obj_class.strip() for obj_class in classes]
    classes = [obj_class for obj_class in classes if obj_class != '']
    
    for c in add_classes:
        if c not in classes:
            classes.append(c)
    
    for c in remove_classes:
        classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
    
    return classes


def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    out = reachable_positions[0]
    min_distance = float('inf')
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out


def euclidean_distance(pos1, pos2):
    # print(pos1)
    # print(pos2)
    return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2 + (pos1['z'] - pos2['z'])**2)

def calculate_object_center(bounding_box):
    x_coords = [point[0] for point in bounding_box]
    y_coords = [point[1] for point in bounding_box]
    z_coords = [point[2] for point in bounding_box]
    center = {
        'x': sum(x_coords) / len(bounding_box),
        'y': sum(y_coords) / len(bounding_box),
        'z': sum(z_coords) / len(bounding_box)
    }
    return center

def find_closest_items(agent_position, scene_graph, num_items=5):
    distances = {}
    for obj_id, obj_data in scene_graph.items():
        # obj_position = obj_data['position']
        obj_aabb = obj_data['BoundingBox']
        obj_center = calculate_object_center(obj_aabb)
        # Adjusting object position relative to the agent
        obj_position_global = {
            'x': obj_center['x'] ,
            'y': obj_center['y'] ,
            'z': obj_center['z'] 
        }
        # Calculate distance from agent to object
        distance = euclidean_distance(agent_position, obj_position_global)
        distances[obj_id] = distance
    # Sort distances and return the closest items
    closest_items = sorted(distances.items(), key=lambda x: x[1])[:num_items]
    return closest_items