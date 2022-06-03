# In this file, we define load_model
# It runs once at server startup to load the model to a GPU
import clip
def load_model():

    # load the model from cache or local file to the CPU
    model,_ = clip.load("ViT-B/32", device="cuda:0")

    # return the callable model
    return model