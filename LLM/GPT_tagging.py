# import base64
# import requests
# import os
# class GPT4ImageClassifier:
#     def __init__(self, api_key, model="gpt-4o-mini"):
#         """
#         Initialize the GPT4ImageClassifier with an API key and model name.
#         """
#         self.api_key = os.environ["OPENAI_API_KEY"]
#         self.model = model
#         self.headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.api_key}"
#         }
    
#     def encode_image(self, image_path):
#         """
#         Encode the image at the given path to a base64 string.
#         """
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
    
#     def generate_payload(self, base64_image, objects_list):
#         """
#         Generate the JSON payload for the API request.
#         """
#         system_prompt = f"Answer what objects are visible in the image. The objects are one of the following: {objects_list}"
        
#         payload = {
#             "model": self.model,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": system_prompt
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": "What’s inside the Fridge?"
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{base64_image}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             "max_tokens": 300
#         }
        
#         return payload
    
#     def classify_image(self, image_path, objects_list):
#         """
#         Encode the image, generate the payload, and send the request to classify the image.
#         """
#         # Encode the image
#         base64_image = self.encode_image(image_path)
        
#         # Generate the payload
#         payload = self.generate_payload(base64_image, objects_list)
        
#         # Send the request
#         response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        
#         # Return the response
#         return response.json()

# # Example usage
# if __name__ == "__main__":
#     # Replace with your OpenAI API key
#     api_key = "YOUR_OPENAI_API_KEY"
    
#     # Create an instance of the classifier
#     classifier = GPT4ImageClassifier(api_key)
    
#     # Define the objects list
#     objects_list = [
#         "Egg|-02.53|+00.60|-00.71", "Mug|+01.45|+00.91|-01.23",
#         "Tomato|+01.30|+00.96|-01.08", "Lettuce|+01.11|+00.83|-01.43",
#         "DishSponge|+01.74|+00.90|-00.86", "Plate|-02.35|+00.90|+00.05",
#         "Potato|-02.24|+00.94|-00.18", "Spatula|-02.31|+00.91|+00.33",
#         "Pot|-02.31|+00.11|+00.89", "Kettle|+00.85|+00.90|-01.79",
#         "SoapBottle|+01.02|+00.90|-01.65", "PaperTowelRoll|+00.69|+01.01|-01.83",
#         "Fork|+01.44|+00.90|+00.34", "SaltShaker|+01.67|+00.90|+00.45",
#         "ButterKnife|+01.44|+00.90|+00.43", "PepperShaker|+01.76|+00.90|+00.37",
#         "Pan|+00.00|+00.90|+00.95", "Knife|-00.64|+00.91|+01.62",
#         "Apple|-00.48|+00.97|+00.41", "Bowl|-00.65|+00.90|+01.26",
#         "Bread|-00.71|+00.98|+00.43", "Cup|-00.65|+00.90|+00.74",
#         "Spoon|-00.66|+00.96|+01.33"
#     ]
    
#     # Path to the image
#     image_path = "/home/hypatia/Sachin_Workspace/interactive-scene-graphs/exploration/Fridge|-02.48|+00.00|-00.78_open.jpg"
    
#     # Classify the image
#     result = classifier.classify_image(image_path, objects_list)
    
#     # Print the result
#     print(result)

from pydantic import BaseModel
from openai import OpenAI
import os
import base64
import instructor

# Define your desired output structure
class ImageClassificationResult(BaseModel):
    detected_objects: bool

# GPT-4 Image Classifier
class GPT4ImageClassifier:
    def __init__(self, api_key, model="gpt-4o-mini"):
        """
        Initialize the GPT4ImageClassifier with an API key and model name.
        """
        # Patch the OpenAI client
        self.client = instructor.from_openai(OpenAI(api_key=api_key))
        self.model = model
    
    def encode_image(self, image_path):
        """
        Encode the image at the given path to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def classify_image(self, image_path, prompt):
        """
        Encode the image, generate the payload, and send the request to classify the image.
        """
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Create the system prompt
        system_prompt = (
            f"First, check if the following objects are visible in the image: {', '.join(contains)}. "
            f"Then, identify any other objects from the following list that are visible: {objects_list}."
        )
        
        # Create the payload
        response = self.client.chat.completions.create(
            model=self.model,
            response_model=ImageClassificationResult,
            messages=[
                # {
                #     "role": "system",
                #     "content": system_prompt,
                # },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,  # General prompt for the image classification
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        # Return the structured result
        return response

# Example usage
if __name__ == "__main__":
    # Retrieve the OpenAI API key from environment variables
    api_key = os.environ["OPENAI_API_KEY"]
    
    # Create an instance of the classifier
    classifier = GPT4ImageClassifier(api_key)
    
    # Define the objects list
    objects_list = [
        "Egg|-02.53|+00.60|-00.71", "Mug|+01.45|+00.91|-01.23",
        "Tomato|+01.30|+00.96|-01.08", "Lettuce|+01.11|+00.83|-01.43",
        "DishSponge|+01.74|+00.90|-00.86", "Plate|-02.35|+00.90|+00.05",
        "Potato|-02.24|+00.94|-00.18", "Spatula|-02.31|+00.91|+00.33",
        "Pot|-02.31|+00.11|+00.89", "Kettle|+00.85|+00.90|-01.79",
        "SoapBottle|+01.02|+00.90|-01.65", "PaperTowelRoll|+00.69|+01.01|-01.83",
        "Fork|+01.44|+00.90|+00.34", "SaltShaker|+01.67|+00.90|+00.45",
        "ButterKnife|+01.44|+00.90|+00.43", "PepperShaker|+01.76|+00.90|+00.37",
        "Pan|+00.00|+00.90|+00.95", "Knife|-00.64|+00.91|+01.62",
        "Apple|-00.48|+00.97|+00.41", "Bowl|-00.65|+00.90|+01.26",
        "Bread|-00.71|+00.98|+00.43", "Cup|-00.65|+00.90|+00.74",
        "Spoon|-00.66|+00.96|+01.33"
    ]
    
    # Path to the image
    image_path = "/home/hypatia/Sachin_Workspace/interactive-scene-graphs/exploration/CounterTop|-01.49|+00.95|+01.32.jpg"
    
    # List of objects you are checking for in the image
    contains = [
        "Pan|+00.00|+00.90|+00.95", "Knife|-00.64|+00.91|+01.62",
        "Apple|-00.48|+00.97|+00.41", "Bowl|-00.65|+00.90|+01.26",
        "Bread|-00.71|+00.98|+00.43", "Cup|-00.65|+00.90|+00.74",
        "Spoon|-00.66|+00.96|+01.33"
    ]
    prompt = "Is the Plate Visible?"
    prompt2 = f"Identify any objects from the following list that are visible: {objects_list}."
    # Classify the image and get the structured result
    result= classifier.classify_image(image_path, prompt)
    # result2 = classifier.classify_image(image_path, prompt2)
    # Print the structured result
    print(result.detected_objects)
    # print(result2.detected_objects)
