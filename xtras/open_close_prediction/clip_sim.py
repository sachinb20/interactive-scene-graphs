import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image1 = preprocess(Image.open("0.jpg")).unsqueeze(0)
image2 = preprocess(Image.open("1.jpg")).unsqueeze(0)
image3 = preprocess(Image.open("5.jpg")).unsqueeze(0)
image4 = preprocess(Image.open("6.jpg")).unsqueeze(0)
text = tokenizer(["close microwave", "open microwave"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features1 = model.encode_image(image1)
    image_features2 = model.encode_image(image2)
    image_features3 = model.encode_image(image3)
    image_features4 = model.encode_image(image4)
    text_features = model.encode_text(text)
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    image_features3 /= image_features3.norm(dim=-1, keepdim=True)
    image_features4 /= image_features4.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs1 = (100.0 * image_features1 @ image_features3.T)
    text_probs2 = (100.0 * image_features1 @ image_features4.T)
    text_probs3 = (100.0 * image_features2 @ image_features3.T)
    text_probs4 = (100.0 * image_features2 @ image_features4.T)
   

print(text_probs1,text_probs2,text_probs3,text_probs4) 
