import os
import torch
import cv2 
from PIL import Image
import sys

if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    

TAG2TEXT_PATH = os.path.join(GSA_PATH, "Tag2Text")
RAM_PATH = os.path.join(GSA_PATH, "recognize-anything")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(RAM_PATH)


try:
    from ram.models.tag2text import tag2text
    from ram.models.ram import ram as rm
    from ram import inference
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e

# Disable torch gradient computation
torch.set_grad_enabled(False)
    
# # GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# # Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")



class TaggingModule:
    def __init__(self, checkpoint_path=TAG2TEXT_CHECKPOINT_PATH):
        """
        Initializes the tagging module by loading the pre-trained model.

        Args:
            checkpoint_path (str, optional): Path to the pre-trained model checkpoint. Defaults to "path/to/model/checkpoint".
        """
        self.delete_tag_index = [i for i in range(3012, 3429)]
        self.specified_tags = 'None'

        # Load model
        self.tagging_model = tag2text(pretrained=checkpoint_path,
                                       image_size=384,
                                       vit='swin_b',
                                       delete_tag_index=self.delete_tag_index)

        # Threshold for tagging
        self.tagging_model.threshold = 0.64

        # Set model to evaluation mode and move to device (CPU or GPU)
        self.tagging_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tagging_model.to(self.device)

        # Define image transformation
        self.tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, bgr_frame):
        """
        Predicts tags and caption for a given BGR frame.

        Args:
            bgr_frame (numpy.ndarray): The BGR image frame.

        Returns:
            tuple: A tuple containing the predicted caption and text prompt.
        """
        # Convert BGR to RGB and create PIL image
        image_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Resize and transform image
        raw_image = image_pil.resize((384, 384))
        raw_image = self.tagging_transform(raw_image).unsqueeze(0).to(self.device)

        # Perform inference
        res = inference.inference_tag2text(raw_image, self.tagging_model, self.specified_tags)
        caption = res[2]
        text_prompt = res[0].replace(' |', ',')

        return caption, text_prompt


