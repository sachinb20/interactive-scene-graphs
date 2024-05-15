import sys
import os
import json
from llama import Llama

class PickAndPlaceAgent:
    def __init__(self):
        LLAMA_PATH = os.environ["LLAMA_PATH"]
        sys.path.append(LLAMA_PATH)

        ckpt_dir = "/home/hypatia/Sachin_Workspace/llama/llama-2-7b-chat"
        tokenizer_path = "/home/hypatia/Sachin_Workspace/llama/tokenizer.model"

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=2000,
            max_batch_size=6,
        )

    def pick_and_place(self, object_list,prompt):
        list_as_string = ', '.join(object_list)
        curr_chat_messages=[
                {"role": "system",
                    "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
                    your objective is to extract specific information from a provided list of objects in the scene. 
                    The Scene Graph represents the environment with objects and receptacles.

                    Your task is to extract two key pieces of information from the Object List:

                    Source Object: Identify the object that needs to be picked up for the task. 
                    Target Receptacle: Determine the receptacle where the source object should be placed. 
                    Ensure that the extracted source object and target receptacle are suitable for executing the pick and place task effectively.

                    The Object List is follows: """ + list_as_string
                },

                # {"role": "user", "content": "Place the apple in fridge please"},
                # {"role": "assistant", "content": """ {"source_object": "Apple","target_receptacle": "Fridge"}"""},

                {"role": "user", "content": "Place the pepper in the cabinet please"},
                {"role": "assistant", "content": """ {"source_object": "PepperShaker","target_receptacle": "Cabinet"}"""},

                {"role": "user", "content": "Wash the potato"},
                {"role": "assistant", "content": """ {"source_object": "Potato","target_receptacle": "Sink"}"""},

                {"role": "user", "content": "Cook something with protein"},
                {"role": "assistant", "content": """ {"source_object": "Egg","target_receptacle": "Pan"}"""},

                {"role": "user", "content": "Clean the mug"},   
                {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "Sink"}"""},

                {"role": "user", "content": "Place the vegetable on the counter"},   
                {"role": "assistant", "content": """ {"source_object": "Lettuce","target_receptacle": "CounterTop"}"""},

                # {"role": "user", "content": "Could you get me some fresh coffee "},
                # {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "CoffeeMachine"}"""}

            ]


        # curr_chat_messages.append({"role": "user", "content": "Could you get me some fresh coffee in a cup"})
        curr_chat_messages.append({"role": "user", "content": prompt})
        curr_chat_messages = [curr_chat_messages]

        chat_completion = self.generator.chat_completion(
            curr_chat_messages,
            max_gen_len=512,
            temperature=0.6,
            top_p=0.9,
        )

        json_object = json.loads(chat_completion[0]['generation']['content'])
        return json_object

if __name__ == "__main__":
    agent = PickAndPlaceAgent()
    json_result = agent.pick_and_place()
    print(json_result)

# import sys
# import os
# import json
# from llama import Llama

# def pick_and_place(object_list):
#     LLAMA_PATH = os.environ["LLAMA_PATH"]
#     sys.path.append(LLAMA_PATH)

#     ckpt_dir = "/home/hypatia/Sachin_Workspace/llama/llama-2-7b-chat"
#     tokenizer_path = "/home/hypatia/Sachin_Workspace/llama/tokenizer.model"

#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=2000,
#         max_batch_size=6,
#     )

#     list_as_string = ', '.join(object_list)

    # curr_chat_messages=[
    #         {
    #             "role": "system",
    #             "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
    #             your objective is to extract specific information from a provided list of objects in the scene. 
    #             The Scene Graph represents the environment with objects and receptacles.

    #             Your task is to extract two key pieces of information from the Object List:

    #             Source Object: Identify the object that needs to be picked up for the task. 
    #             Target Receptacle: Determine the receptacle where the source object should be placed. 
    #             Ensure that the extracted source object and target receptacle are suitable for executing the pick and place task effectively.

    #             The Object List is follows: """ + list_as_string
    #         },

    #         # {"role": "user", "content": "Place the apple in fridge please"},
    #         # {"role": "assistant", "content": """ {"source_object": "Apple","target_receptacle": "Fridge"}"""},

    #         {"role": "user", "content": "Place the pepper in the cabinet please"},
    #         {"role": "assistant", "content": """ {"source_object": "PepperShaker","target_receptacle": "Cabinet"}"""},

    #         {"role": "user", "content": "Wash the potato"},
    #         {"role": "assistant", "content": """ {"source_object": "Potato","target_receptacle": "Sink"}"""},

    #         {"role": "user", "content": "Cook something with protein"},
    #         {"role": "assistant", "content": """ {"source_object": "Egg","target_receptacle": "Pan"}"""},

    #         {"role": "user", "content": "Clean the mug"},   
    #         {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "Sink"}"""},

    #         {"role": "user", "content": "Place the vegetable on the counter"},   
    #         {"role": "assistant", "content": """ {"source_object": "Lettuce","target_receptacle": "CounterTop"}"""},

    #         # {"role": "user", "content": "Could you get me some fresh coffee "},
    #         # {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "CoffeeMachine"}"""}

    #     ]


    # curr_chat_messages.append({"role": "user", "content": "Could you get me some fresh coffee in a cup"})
    # # curr_chat_messages.append({"role": "user", "content": "I'd like to have my apple chilled. Could you find a cool place to keep it."})
    # curr_chat_messages = [curr_chat_messages]

#     chat_completion = generator.chat_completion(
#         curr_chat_messages,
#         max_gen_len=512,
#         temperature=0.6,
#         top_p=0.9,
#     )

#     json_object = json.loads(chat_completion[0]['generation']['content'])
#     return json_object

# if __name__ == "__main__":
#     json_result = pick_and_place()
#     print(json_result)

# import sys
# import os
# LLAMA_PATH = os.environ["LLAMA_PATH"]
# sys.path.append(LLAMA_PATH)

# from llama import Llama, Dialog


# # LLAMA3_PATH = os.environ["LLAMA3_PATH"]
# # sys.path.append(LLAMA3_PATH)

# # from llama import Llama, Dialog


# from typing import List
# import json



# ckpt_dir = "/home/hypatia/Sachin_Workspace/llama/llama-2-7b-chat"
# tokenizer_path = "/home/hypatia/Sachin_Workspace/llama/tokenizer.model"

# # ckpt_dir = "/home/hypatia/Sachin_Workspace/llama3/Meta-Llama-3-8B-Instruct"
# # tokenizer_path = "/home/hypatia/Sachin_Workspace/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"

# generator = Llama.build(
#     ckpt_dir=ckpt_dir,
#     tokenizer_path=tokenizer_path,
#     # model_parallel_size = 1,
#     max_seq_len=2000,
#     max_batch_size=6,
# )

# file_path = 'object_list.txt'

# # Open the file in read mode
# with open(file_path, 'r') as file:
#     # Read the contents of the file and save it as a string
#     file_content = file.read()

# curr_chat_messages=[
#         {
#             "role": "system",
#             "content": """Given the context of being a Planning Agent for a Robotics Pick and Place task, 
#             your objective is to extract specific information from a provided list of objects in the scene. 
#             The Scene Graph represents the environment with objects and receptacles.

#             Your task is to extract two key pieces of information from the Object List:

#             Source Object: Identify the object that needs to be picked up for the task. 
#             Target Receptacle: Determine the receptacle where the source object should be placed. 
#             Ensure that the extracted source object and target receptacle are suitable for executing the pick and place task effectively.

#             The Object List is follows: """ + file_content
#         },

#         # {"role": "user", "content": "Place the apple in fridge please"},
#         # {"role": "assistant", "content": """ {"source_object": "Apple","target_receptacle": "Fridge"}"""},

#         {"role": "user", "content": "Place the pepper in the cabinet please"},
#         {"role": "assistant", "content": """ {"source_object": "PepperShaker","target_receptacle": "Cabinet"}"""},

#         {"role": "user", "content": "Wash the potato"},
#         {"role": "assistant", "content": """ {"source_object": "Potato","target_receptacle": "Sink"}"""},

#         {"role": "user", "content": "Cook something with protein"},
#         {"role": "assistant", "content": """ {"source_object": "Egg","target_receptacle": "Pan"}"""},

#         {"role": "user", "content": "Clean the mug"},   
#         {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "Sink"}"""},

#         {"role": "user", "content": "Place the vegetable on the counter"},   
#         {"role": "assistant", "content": """ {"source_object": "Lettuce","target_receptacle": "CounterTop"}"""},

#         # {"role": "user", "content": "Could you get me some fresh coffee "},
#         # {"role": "assistant", "content": """ {"source_object": "Mug","target_receptacle": "CoffeeMachine"}"""}

#     ]

# curr_chat_messages.append({"role": "user", "content": "Could you get me some fresh coffee in a cup"})
# # curr_chat_messages.append({"role": "user", "content": "I'd like to have my apple chilled. Could you find a cool place to keep it."})
# curr_chat_messages = [curr_chat_messages]

# chat_completion = generator.chat_completion(
#     curr_chat_messages,  # type: ignore
#     max_gen_len=512,
#     temperature=0.6,
#     top_p=0.9,
# )


# json_object = json.loads(chat_completion[0]['generation']['content'])

# # Now json_object contains the JSON object
# print(json_object["source_object"])