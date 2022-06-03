from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def encode_text(text,model):
    tokens = clip.tokenize([text]).to("cuda:0")
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    return text_features

def encode_image(image_bytes,model):
    transform = Compose([
        Resize(model.visual.input_resolution, interpolation=BICUBIC),
        CenterCrop(model.visual.input_resolution),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    pil_image = Image.open(image_bytes)
    image_size = pil_image.size
    image = transform(pil_image).unsqueeze(0).to("cuda:0")

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features

def get_similarity(text_encoding, image_encoding):

    dot_p = torch.dot(text_encoding[0], image_encoding[0].T)

    text_norm = torch.norm(text_encoding[0])
    image_norm = torch.norm(image_encoding[0])

    norms = text_norm * image_norm

    similarity = dot_p.cpu().numpy() / norms.cpu().numpy()
    return similarity.item()