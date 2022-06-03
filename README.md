
# serverless-template for CLIP

Setup and host CLIP on Banana in minutes

# How to use this repo:

1) Fork this repo

2) Tweak the repo to your liking:
- `requirements.txt` 
	- this file holds the pip dependencies, which are installed via the Dockerfile.
	- add or remove your pip packages, one per line.
- `src/download.py` 
	- this file downloads your model weights to the local file system during the build step. 
	- This is an optional file. The only goal is to get your model weights built into the docker image. This example uses a `download.py` script to download weights. If you can acheive the download through other means, such as a `RUN cURL ...` from the Dockerfile, feel free.
- `src/warmup.py` 
	- this file defines `load_model()`, which loads the model from local weights, loads it onto the GPU, and returns the model object.
	- add or remove logic to the `load_model()` function for any logic that you want to run once at system startup, before the http server starts.
	- the max size of a model is currently limited to 15gb in GPU RAM. Banana does not support greater than that at the moment.
- `src/run.py` 
	- this file defines ML related logic for each call including tensor preprocessing, sampling logic in postprocessing, etc.
- `src/app.py`
	- this file defines the http server (Sanic, in this case, but you can change to Flask) which starts once the load_model() finishes.
	- edit this file to define the API
		- the values you parse from model_inputs defines the JSON schema you'd use as inputs
		- the json you return as model_outputs defines the JSON schema you'd expect as an output

3) Test and verify it works

# Deploying to Banana hosted Serverless GPUs:

1) [Log into the Banana dashboard](https://app.banana.dev/) and get your API Key

2) Email us at `onboarding@banana.dev` with the following message:
```
Hello, I'd like to be onboarded to serverless.
My github username is: YOUR_GITHUB_USERNAME
My Banana API Key is: YOUR_API_KEY
My preferred billing email is: YOU@EMAIL.COM
```
Your github username, banana api key, and email are required for us to authorize you into the system. 
We will reply and confirm when you're added.

3) Install the [Banana Github App](https://github.com/apps/banana-serverless) to the forked repo. 

4) Push to main to trigger a build and deploy to the hosted backend. This will do nothing if you have not completed the email in step 2 above.

6) Monitor your email inbox for status updates, which will include a unique model key for this repo for you to use through the Banana SDKs.

To continue to the Banana SDKs, find them linked here:
- [Python](https://github.com/bananaml/banana-python-sdk)
- [Node JS / Typescript](https://github.com/bananaml/banana-node-sdk)
- [Go](https://github.com/bananaml/banana-go)
