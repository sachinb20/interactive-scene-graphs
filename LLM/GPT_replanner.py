import os
import json
import openai
import instructor
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Action(BaseModel):
    action: str
    object: str
    target: Optional[str] = None

class ActionSequence(BaseModel):
    actions: List[Action]

class ReplanningAgent:
    def __init__(self):
        # Set your OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")
        openai.api_key = self.api_key
        # Patch the OpenAI client with instructor
        self.client = instructor.from_openai(openai.OpenAI())

    def replanner(self, embodiment: str, scene_graph: Dict[str, Any], task: str, action_history: List[Action], feedback: str, temperature:float):
        curr_chat_messages = [
            {
                "role": "system",
                "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
                your objective is to generate a sequence of actions based on a provided list of objects in the scene and
                any feedback provided. You need to consider the history of actions taken, the current state of the scene graph,
                and the feedback to generate a revised sequence of actions to complete the task.

                The Scene Graph represents the environment with objects and receptacles. Feedback might indicate that
                an object is not in the correct place or state, or that new objects have been added to the scene graph. 
                Your task is to determine the actions required to address these issues and complete the task. 
                Actions can include navigating to the object, picking it up, placing it, opening or closing receptacles if needed.

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
                Imprortant: Pick and Place actions must be specified with the Target receptacle and must not be none
                                
                After finishing the planning sequence the robot must be set on standby.
                The following actions are available:
                Action_set: [navigate,pick_object,put_object,open_receptacle,close_receptacle,standby]  

                Task: Description of Task

                Action_History: List of tasks that have been taken so far

                Feedback: Description of feedback received
                """
            },
            {
                "role": "user", 
                "content": f"Scene_Graph: {json.dumps(scene_graph)} Task: {task} Action_History: {json.dumps(action_history)} Feedback: {feedback}"
            }
        ]

        try:
            # Use the instructor-patched client to generate structured data
            response = self.client.chat.completions.create(
                model="gpt-4",
                response_model=ActionSequence,
                messages=curr_chat_messages,
                max_tokens=1024,
                temperature=float(temperature),
                top_p=0.9,
            )
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

if __name__ == "__main__":
    replanning_agent = ReplanningAgent()

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
        if task == "task1":
            continue
        print(f"Task: {task}")
        # for prompt in details['prompts']:
        #     print(f"Prompt: {prompt}")

        task = details['prompts'][2]
        embodiment = "Bi-manipulation"
        action_history = details['prompts'][:2]
        feedback = "The Fork is not on the CounterTop"
        temperature = 0.6

        sg = {'Agent': {'contains': [], 'State': 'clear'}, 'Fridge|-02.48|+00.00|-00.78': {'contains': [], 'State': 'Closed'}, 'Sink|+01.38|+00.81|-01.27': {'contains': ['Tomato|+01.30|+00.96|-01.08'], 'State': 'clear'}, 'Microwave|-02.58|+00.90|+02.44': {'contains': [], 'State': 'Closed'}, 'CounterTop|+01.59|+00.95|+00.41': {'contains': ['Fork|+01.44|+00.90|+00.34', 'SaltShaker|+01.67|+00.90|+00.45', 'ButterKnife|+01.44|+00.90|+00.43'], 'State': 'clear'}, 'CounterTop|-00.36|+00.95|+01.09': {'contains': ['Apple|-00.48|+00.97|+00.41', 'Bowl|-00.65|+00.90|+01.26', 'Cup|-00.65|+00.90|+00.74'], 'State': 'clear'}, 'Drawer|-02.28|+00.79|+01.37': {'contains': [], 'State': 'Closed'}, 'Cabinet|+00.15|+02.01|-01.60': {'contains': [], 'State': 'Closed'}}
        task = "Cook the Apple in the Microwave"
        action_history = ['Put the Apple in the Sink', 'Open and Close the Cabinet']
        feedback = "The Apple|-00.48|+00.97|+00.41 is not on/in the CounterTop|-00.36|+00.95|+01.09. Find another place to pick the apple"
        print(task)
        print(action_history)

        action_sequence = replanning_agent.replanner(embodiment, sg, task, action_history, feedback, temperature)
    
        # Print the structured output
        if action_sequence:
            for action in action_sequence.actions:
                print(f"Action: {action.action}, Object: {action.object}, Target: {action.target}")
        else:
            print("Failed to generate action sequence.")


# import os
# import json
# import openai

# class PickAndPlaceAgent:
#     def __init__(self):
#         # Set your OpenAI API key
#         self.api_key = os.environ["OPENAI_API_KEY"]
#         openai.api_key = self.api_key

#     def re_planner(self, embodiment, scene_graph, task):
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

#     prev_plan = [
#                     {"action": "navigate", "object": "Lettuce"},
#                     {"action": "pick", "object": "Lettuce"},
#                     {"action": "navigate", "object": "Sink"},
#                     {"action": "place", "object": "Lettuce", "target": "Sink"},
#                     {"action": "navigate", "object": "Tomato"},
#                     {"action": "pick", "object": "Tomato"},
#                     {"action": "navigate", "object": "Sink"},
#                     {"action": "place", "object": "Tomato", "target": "Sink"},
#                     {"action": "navigate", "object": "Fridge"},
#                     {"action": "open", "object": "Fridge"},
#                     {"action": "navigate", "object": "Sink"},
#                     {"action": "pick", "object": "Lettuce", "target": "Sink"},
#                     {"action": "navigate", "object": "Fridge"},
#                     {"action": "place", "object": "Lettuce", "target": "Fridge"},
#                     {"action": "navigate", "object": "Sink"},
#                     {"action": "pick", "object": "Tomato", "target": "Sink"},
#                     {"action": "navigate", "object": "Fridge"},
#                     {"action": "place", "object": "Tomato", "target": "Fridge"},
#                     {"action": "close", "object": "Fridge"}
#                 ]


#     task = "Wash all the vegetables and then store them in the fridge."
#     feedback = "The "

#     json_result = agent.re_planner(data, )
#     print(json_result)

