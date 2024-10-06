import os
import json
import openai
import instructor
from pydantic import BaseModel
from typing import Optional, List

# Define the Pydantic models as before
class Action(BaseModel):
    action: str
    object: str
    target: Optional[str] = None

class ActionSequence(BaseModel):
    actions: List[Action]

class PlanningAgent:
    def __init__(self):
        # Set your OpenAI API key
        self.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.api_key
        # Patch the OpenAI client with instructor
        self.client = instructor.from_openai(openai.OpenAI())

    def planner(self, embodiment, scene_graph, task):
        curr_chat_messages = [
            {
                "role": "system",
                "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
                your objective is to generate a sequence of actions based on a provided list of objects in the scene. 
                The Scene Graph represents the environment with objects and receptacles.

                Your task is to determine the actions required to complete the given task. 
                These actions can include navigating to the object, picking it up, placing it, opening or closing receptacles if needed.

                The Prompt format is as follows: 
                Scene_Graph:{
                                "Receptacle": 
                                {   "state": "state", 
                                    "contains": [
                                        {"object": "ContainedObject|Position", "state": "state"},
                                        {"object": "ContainedObject|Position", "state": "state"}]
                                },
                                "Receptacle|Position": 
                                {   "state": "state", 
                                    "contains": [
                                        {"object": "ContainedObject|Position", "state": "state"},
                                        {"object": "ContainedObject|Position", "state": "state"}]
                                }
                            }

                Important: Only Receptacle can be navigated to, you cannot navigate directly to an object
                Important: To close, open an object you must be near it
                Important: When picking and placing an object the target must not be None

                After finishing the planning sequence the robot must be set on standby.
                The following actions are available:
                Action_set: [navigate,pick_object,put_object,open_receptacle,close_receptacle,standby]  

                Task: Description of Task
                """
            },
            {
                "role": "user", 
                "content": f"Scene_Graph: {json.dumps(scene_graph)} Task: {task}"
            }
        ]

        # Use the instructor-patched client to generate structured data
        response = self.client.chat.completions.create(
            model="gpt-4",
            response_model=ActionSequence,
            messages=curr_chat_messages,
            max_tokens=1024,
            temperature=0.6,
            top_p=0.9,
        )

        # Return the structured action sequence
        return response

if __name__ == "__main__":
    agent = PlanningAgent()

    # Load the JSON file
    with open("/home/hypatia/Sachin_Workspace/interactive-scene-graphs/sg_data/FloorPlan6_sg.json", 'r') as file:
        data = json.load(file)

    sg = {}
    for key, value in data.items():
        sg[key] = {
            "contains": value["contains"],
            "State": value["state"]
        }


    scene = "FloorPlan6"
    experiment_type = "multi_step"
    file_name = f"../tasks/{scene}/{experiment_type}.json"

    # Open and read the JSON file
    with open(file_name, 'r') as file:
        json_data = json.load(file)

    # Iterate through the JSON data and print each prompt
    for task, details in json_data.items():
        print(f"Task: {task}")
        for prompt in details['prompts']:
            print(f"Prompt: {prompt}")

    # prompt = "Put the Lettuce in the Fridge."
            embodiment = "Bi-manipulation"
            action_sequence = agent.planner(embodiment, sg, prompt)
            
            # Print the structured output
            for action in action_sequence.actions:
                print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")


# import os
# import json
# import openai

# class PickAndPlaceAgent:
#     def __init__(self):
#         # Set your OpenAI API key
#         self.api_key = os.environ["OPENAI_API_KEY"]
#         openai.api_key = self.api_key

#     def pick_and_place(self, embodiment, scene_graph, task):
#         curr_chat_messages = [
#             {
#                 "role": "system",
#                 "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
#                 your objective is to generate a sequence of actions based on a provided list of objects in the scene. 
#                 The Scene Graph represents the environment with objects and receptacles.

#                 Your task is to determine the actions required to complete the given task. 
#                 These actions can include navigating to the object, picking it up, placing it, opening or closing receptacles if needed.

#                 The Prompt format is as follows: 
#                 Scene_Graph:{
#                                 "Receptacle": 
#                                 {   "state": "state", 
#                                     "contains": [
#                                         {"object": "ContainedObject|Position", "state": "state"},
#                                         {"object": "ContainedObject|Position", "state": "state"}]
#                                 },
#                                 "Receptacle|Position": 
#                                 {   "state": "state", 
#                                     "contains": [
#                                         {"object": "ContainedObject|Position", "state": "state"},
#                                         {"object": "ContainedObject|Position", "state": "state"}]
#                                 }
#                             }
#                 The following actions are available:
#                 Action_set: [navigate,pick,place,open,close]  

#                 Task: Description of Task
#                 """
#             },
#             {
#                 "role": "user", 
#                 "content": """Scene_Graph: {
#                     "Drawer|+00.5|+00.9|+00.5":
#                     {"state": "closed", 
#                     "contains": [
#                         {"object": "PepperShaker|+00.5|+00.9|+00.6", "state": "inside"},
#                         {"object": "Knife|+00.6|+00.9|+00.6", "state": "inside"}
#                     ]}
#                     "Cabinet|-02.15|+00.40|+00.70": 
#                     {"state": "closed", 
#                     "contains": [
#                         {"object": "Bowl|-00.65|+00.90|+01.26", "state": "inside"},
#                         {"object": "SmallBowl|-00.70|+00.90|+01.20", "state": "inside"}
#                     ]}
#                 }
#                 Task: Place the pepper in the cabinet please"""
#             },
#             {
#                 "role": "assistant",
#                 "content": """[
#                     {"action": "navigate", "object": "Drawer"},
#                     {"action": "open", "object": "Drawer"},
#                     {"action": "pick", "object": "PepperShaker", "target": "Drawer"},
#                     {"action": "close", "object": "Drawer"},
#                     {"action": "navigate", "object": "Cabinet"},
#                     {"action": "open", "object": "Cabinet"},
#                     {"action": "place", "object": "PepperShaker", "target": "Cabinet"},
#                     {"action": "close", "object": "Cabinet"}
#                 ]"""
#             }
#         ]

#         prompt = "Scene_Graph: " + str(scene_graph) + " Task: " + task
#         curr_chat_messages.append({"role": "user", "content": prompt})

#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=curr_chat_messages,
#             max_tokens=512,
#             temperature=0.6,
#             top_p=0.9,
#         )

#         chat_completion = response.choices[0].message.content
#         print(chat_completion)

#         # json_object = json.loads(chat_completion)
#         return 0

# if __name__ == "__main__":
#     agent = PickAndPlaceAgent()

#     # Load the JSON file
#     with open('reversed_scene_graph_compressed.json', 'r') as file:
#         data = json.load(file)

#     task = "Wash all the vegetables and then store them in the fridge."
#     embodiment = "Bi-manipulation"
#     json_result = agent.pick_and_place(embodiment, data, task)
#     print(json_result)

