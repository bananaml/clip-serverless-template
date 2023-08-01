from potassium import Potassium, Request, Response
from transformers import CLIPProcessor,CLIPModel,CLIPTokenizerFast
import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image


app = Potassium("my_app")


@app.init
def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model name/id
    model_id = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    
    context = {
        "model": model,
        "processor": processor,
        "tokenizer": tokenizer,
        
    }

    return context


# @app.handler is an http post handler running for every call
@app.handler("/text")
def handler(context: dict, request: Request) -> Response:
    
    prompt = request.json.get("prompt")
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.get_text_features(**inputs)
    out = out.squeeze(0)
    emb = out.cpu().detach().numpy().tolist()

    return Response(
        json = {"outputs": emb}, 
        status=200
    )

@app.handler("/image")
def handler(context: dict, request: Request) -> Response:
    
    image_url = request.json.get("imageURL")
    if image_url != None:
        pil_image = Image.open(requests.get(image_url, stream=True).raw)
    else:
        assert False, "No image provided"
    
    model = context.get("model")
    processor = context.get("processor")
    image = processor(text=None,
                  images=pil_image,
                  return_tensors='pt',
                  padding=True
                 )['pixel_values']
    out = model.get_image_features(pixel_values=image)
    out = out.squeeze(0)
    emb = out.cpu().detach().numpy()
    emb = emb/np.linalg.norm(emb)

    return Response(
        json = {"outputs": emb}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()


# @app.route('/healthcheck', methods=["GET"])
# def healthcheck(request):
#     return response.json({"state": "healthy"})

# @app.route('/', methods=["POST"]) # Do not edit - POST requests to "/" are a required interface
# def inference(request):
#     try:
#         model_parameters = json.loads(request.json)
#     except:
#         model_parameters = request.json
        
#     image_byte_string = model_parameters.get('imageByteString', None)
#     image_url = model_parameters.get('imageURL', None)
#     text = model_parameters.get('text', None)
#     texts = model_parameters.get('texts', None)

#     if image_byte_string == None and image_url == None:
#         return json({'message': "No image provided"})

#     if text == None and texts ==  None:
#         return json({'message': "No text provided"})

    
#     image_bytes = None

#     if image_url != None:
#         response = requests.get(image_url)
#         image_bytes = BytesIO(response.content)

#     if image_byte_string != None:
#         image_encoded = image_byte_string.encode('utf-8')
#         image_bytes = BytesIO(base64.b64decode(image_encoded))

#     # preencode the image
#     image_encoding = encode_image(image_bytes,model)

#     response = {}

#     if texts != None:
#         sims = []
#         for t in texts:
#             text_encoding = encode_text(t,model)
#             sim = get_similarity(text_encoding, image_encoding)
#             sims.append(sim)
#             response["similarities"] = sims

#     if text != None:
#         text_encoding = encode_text(text,model)
#         sim = get_similarity(text_encoding, image_encoding)
#         response['similarity'] = sim

#     return json(response) # Do not edit - returning a dictionary as JSON is a required interface


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000, workers=1)