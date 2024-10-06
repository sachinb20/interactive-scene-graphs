import sys
import os
import json
from llama import Llama

class PickAndPlaceAgent:
    def __init__(self):
        LLAMA_PATH = os.environ["LLAMA3_PATH"]
        sys.path.append(LLAMA_PATH)

        ckpt_dir = LLAMA_PATH+"/Meta-Llama-3-8B-Instruct"
        tokenizer_path = LLAMA_PATH+"/Meta-Llama-3-8B-Instruct/tokenizer.model"
        # LLAMA_PATH = os.environ["LLAMA_PATH"]
        # sys.path.append(LLAMA_PATH)

        # ckpt_dir = "/home/hypatia/Sachin_Workspace/llama/llama-2-7b-chat"
        # tokenizer_path = "/home/hypatia/Sachin_Workspace/llama/tokenizer.model"

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=4000,
            max_batch_size=6,
        )

    def pick_and_place(self, scene_graph,task):
        
        curr_chat_messages=[
                            {"role": "system",
                                "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
                                your objective is to generate a sequence of actions based on a provided list of objects in the scene. 
                                The Scene Graph represents the environment with objects and receptacles.

                                Your task is to determine the actions required to complete the given task. 
                                These actions can include navigating to the object, picking it up, placing it, opening or closing receptacles if needed.

                                The Prompt format is as follows: 
                                Scene_Graph:[{"object": "Receptacle", "state": "state", "contains": [
                                                {"object": "ContainedObject|Position", "state": "state"},
                                                {"object": "ContainedObject|Position", "state": "state"}
                                            ]},
                                            {"object": "Receptacle|Position", "state": "state", "contains": [
                                                {"object": "ContainedObject|Position", "state": "state"},
                                                {"object": "ContainedObject|Position", "state": "state"}
                                            ]}
                                            ]
                                Task: Description of Task
                                """},


                                {"role": "user", "content": """Scene_Graph: [
                                    {"object": "Drawer|+00.5|+00.9|+00.5", "state": "closed", "contains": [
                                        {"object": "PepperShaker|+00.5|+00.9|+00.6", "state": "inside"},
                                        {"object": "Knife|+00.6|+00.9|+00.6", "state": "inside"}
                                    ]},
                                    {"object": "Cabinet|-02.15|+00.40|+00.70", "state": "closed", "contains": [
                                        {"object": "Bowl|-00.65|+00.90|+01.26", "state": "inside"},
                                        {"object": "SmallBowl|-00.70|+00.90|+01.20", "state": "inside"}
                                    ]}
                                ]
                                Task: Place the pepper in the cabinet please"""},

                                {"role": "assistant", "content": """[
                                    {"action": "navigate", "object": "Drawer"},
                                    {"action": "open", "object": "Drawer"},
                                    {"action": "pick", "object": "PepperShaker"},
                                    {"action": "close", "object": "Drawer"},
                                    {"action": "navigate", "object": "Cabinet"},
                                    {"action": "open", "object": "Cabinet"},
                                    {"action": "place", "object": "PepperShaker", "target": "Cabinet"},
                                    {"action": "close", "object": "Cabinet"}
                                ]"""},

                                {"role": "user", "content": """Scene_Graph: [
                                    {"object": "Pantry|+00.2|+00.5|+00.3", "state": "closed", "contains": [
                                        {"object": "Potato|+00.3|+00.5|+00.3", "state": "inside"},
                                        {"object": "Onion|+00.4|+00.5|+00.3", "state": "inside"}
                                    ]},
                                    {"object": "Sink|-01.2|+00.6|+00.8", "state": "clear", "contains": []}
                                ]
                                Task: Wash the potato"""},

                                {"role": "assistant", "content": """[
                                    {"action": "navigate", "object": "Pantry"},
                                    {"action": "open", "object": "Pantry"},
                                    {"action": "pick", "object": "Potato"},
                                    {"action": "close", "object": "Pantry"},
                                    {"action": "navigate", "object": "Sink"},
                                    {"action": "place", "object": "Potato", "target": "Sink"}
                                ]"""},

                                {"role": "user", "content": """Scene_Graph: [
                                    {"object": "Fridge|-00.9|+00.3|+00.2", "state": "closed", "contains": [
                                        {"object": "Egg|-00.8|+00.3|+00.2", "state": "inside"},
                                        {"object": "Cheese|-00.7|+00.3|+00.2", "state": "inside"}
                                    ]},
                                    {"object": "Pan|+00.6|+00.9|+00.5", "state": "clear", "contains": []}
                                ]
                                Task: Cook something with protein"""},

                                {"role": "assistant", "content": """[
                                    {"action": "navigate", "object": "Fridge"},
                                    {"action": "open", "object": "Fridge"},
                                    {"action": "pick", "object": "Egg"},
                                    {"action": "close", "object": "Fridge"},
                                    {"action": "navigate", "object": "Pan"},
                                    {"action": "place", "object": "Egg", "target": "Pan"}
                                ]"""},

                                {"role": "user", "content": """Scene_Graph: [
                                    {"object": "Table|+01.0|+00.5|+00.5", "state": "clear", "contains": [
                                        {"object": "Mug|+01.1|+00.5|+00.5", "state": "on"},
                                        {"object": "Plate|+01.2|+00.5|+00.5", "state": "on"}
                                    ]},
                                    {"object": "Sink|-01.2|+00.6|+00.8", "state": "clear", "contains": []}
                                ]
                                Task: Clean the mug"""},

                                {"role": "assistant", "content": """[
                                    {"action": "navigate", "object": "Table"},
                                    {"action": "pick", "object": "Mug"},
                                    {"action": "navigate", "object": "Sink"},
                                    {"action": "place", "object": "Mug", "target": "Sink"}
                                ]"""},

                                {"role": "user", "content": """Scene_Graph: [
                                    {"object": "Fridge|-00.9|+00.3|+00.2", "state": "closed", "contains": [
                                        {"object": "Lettuce|-00.8|+00.3|+00.2", "state": "inside"},
                                        {"object": "Carrot|-00.7|+00.3|+00.2", "state": "inside"}
                                    ]},
                                    {"object": "CounterTop|+00.5|+00.9|+00.5", "state": "clear", "contains": []}
                                ]
                                Task: Place the vegetable on the counter"""},

                                {"role": "assistant", "content": """[
                                    {"action": "navigate", "object": "Fridge"},
                                    {"action": "open", "object": "Fridge"},
                                    {"action": "pick", "object": "Lettuce"},
                                    {"action": "close", "object": "Fridge"},
                                    {"action": "navigate", "object": "CounterTop"},
                                    {"action": "place", "object": "Lettuce", "target": "CounterTop"}
                                ]"""},
                            ]


        # curr_chat_messages.append({"role": "user", "content": "Could you get me some fresh coffee in a cup"})
        prompt = "Scene_Graph: "+str(scene_graph)+" Task: "+task
        curr_chat_messages.append({"role": "user", "content": prompt})
        curr_chat_messages = [curr_chat_messages]

        chat_completion = self.generator.chat_completion(
            curr_chat_messages,
            max_gen_len=512,
            temperature=0.6,
            top_p=0.9,
        )
        print(chat_completion)

        json_object = json.loads(chat_completion[0]['generation']['content'])
        return json_object

if __name__ == "__main__":
    agent = PickAndPlaceAgent()

    # Load the JSON file
    with open('reversed_scene_graph_compressed.json', 'r') as file:
        data = json.load(file)

    # Extract all keys into a list
    # keys_list = list(data.keys())

    json_result = agent.pick_and_place(data,"I'd like to have my apple heated.")
    print(json_result)
