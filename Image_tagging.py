import os
import torch
import cv2 
from PIL import Image
import sys
# Disable torch gradient computation
torch.set_grad_enabled(False)
    


##################### DETIC ################
DETIC_PATH = os.environ["DETIC_PATH"]
CENTERNET_PATH = os.path.join(DETIC_PATH, "third_party/CenterNet2/")
sys.path.append(DETIC_PATH)
sys.path.insert(0,CENTERNET_PATH)

DETECTRON_PATH = os.environ["DETECTRON_PATH"]
sys.path.append(DETECTRON_PATH)

from centernet.config import add_centernet_config

#TODO:Reference to detectron and detic paths
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from detic.modeling.utils import reset_cls_test
from detic.predictor import AsyncPredictor
from detic.modeling.text.text_encoder import build_text_encoder
from detic.config import add_detic_config



BUILDIN_CLASSIFIER = {
    'lvis': '/home/hypatia/Sachin_Workspace/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}
##################################################

class Detic:
    def __init__(self, instance_mode=ColorMode.IMAGE, confidence_threshold=0.5, DEVICE='gpu'):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        # cfg = setup_cfg(args)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.pred_all_class = True
        self.config_file = DETIC_PATH + "/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        self.opts = ['MODEL.WEIGHTS', DETIC_PATH + '/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth']

        cfg = get_cfg()
        if DEVICE == 'cpu':
            cfg.MODEL.DEVICE="cpu"
        add_centernet_config(DETIC_PATH,cfg)
        add_detic_config(DETIC_PATH,cfg)
        cfg.merge_from_file(self.config_file)
        cfg.merge_from_list(self.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
        if not self.pred_all_class:
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()

        self.config = cfg



    def tag(self, vocabulary, custom_vocabulary, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        image = read_image(image, format="BGR")

        if vocabulary == 'custom':
            metadata = MetadataCatalog.get("__unused")
            metadata.thing_classes = custom_vocabulary.split(',')
            classifier = get_clip_embeddings(metadata.thing_classes)
        else:
            metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]


        num_classes = len(metadata.thing_classes)
        predictor = DefaultPredictor(self.config)
        reset_cls_test(predictor.model, classifier, num_classes)


        vis_output = None
        predictions = predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
        if vocabulary == 'custom':
            del metadata.thing_classes      
        return predictions, vis_output


##################### LLAVA ################
LLAVA_PYTHON_PATH = os.environ["LLAVA_PYTHON_PATH"]
sys.path.append(LLAVA_PYTHON_PATH)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
##############################################

class LLAVA:
    def __init__(self):
        self.model_path = os.getenv("LLAVA_CKPT_PATH")
        self.conv_mode = "v0_mmtag" # "multimodal"
        self.num_gpus = 1



    def tag(self,query,image_file):

        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": query,
            "conv_mode":  self.conv_mode,
            "image_file": image_file,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        output = eval_model(args)
        return output



def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


# # GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
# GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

# # Segment-Anything checkpoint
# SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")


##################### RAM and Tag2text ################
if "GSA_PATH" in os.environ:
    GSA_PATH = os.environ["GSA_PATH"]
else:
    raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
    

TAG2TEXT_PATH = os.path.join(GSA_PATH, "Tag2Text")
RAM_PATH = os.path.join(GSA_PATH, "recognize-anything")
sys.path.append(GSA_PATH) # This is needed for the following imports in this file
sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
sys.path.append(RAM_PATH)
# Tag2Text checkpoint
TAG2TEXT_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./tag2text_swin_14m.pth")
RAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./ram_swin_large_14m.pth")

try:
    from ram.models.tag2text import tag2text
    from ram.models.ram import ram as rm
    from ram import inference
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your GSA_PATH. ")
    raise e
####################################################

class RAM:
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




# tag = Detic()

# # Directory containing the images
# image_directory = 'exploration_images'
# output_directory = 'exploration_images/output_images'

# # Make sure the output directory exists
# os.makedirs(output_directory, exist_ok=True)

# # List all .jpg image files in the directory
# image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

# # Iterate over all image files
# for image_file in image_files:
#     image_path = os.path.join(image_directory, image_file)
    
#     # Run the detection
#     predictions, vis_output = tag.tag("custom", "MIcrowave", image_path)
    
#     # Save the visual output
#     output_path = os.path.join(output_directory, f"output_{image_file}")
#     vis_output.save(output_path)
    
#     # Print the predictions
#     print(f"Predictions for {image_file}: {predictions}")



# tagging_module = LLAVA()
# query = "Is the Drawer door open or closed? Answer in a single word"
# image_file = f"open_close_prediction/Drawers_Open/image_2.jpg"
# op = tagging_module.tag(query,image_file)
# print(op)
