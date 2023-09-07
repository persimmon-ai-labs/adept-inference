Persimmon-8B User Guide
==========
This repo contains inference code for [Persimmon-8B](https://www.adept.ai/blog/persimmon-8b), the new LLM from Adept.

Downloading the Checkpoint
--------

The model checkpoints are stored on our public OCI bucket and can be downloaded using `wget`.
The base model is not fine-tuned and is released under an Apache 2.0 license.
The chat model is fine-tuned and is released under a CC-BY-NC 4.0 license.

Base:
https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar

Chat:
https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar

Untar the model into its own directory via `tar -xvf 8b_base_model_release.tar` or `tar -xvf 8b_chat_model_release.tar`

The scripts are set up to expect the model folder to be placed within the code directory, but you can place it elsewhere and modify the scripts accordingly.

Building Docker
-----------

Build the docker that will include all the necessary dependencies (and then some!) using the included Dockerfile:

```
docker build -f docker/Dockerfile -t 'adeptdocker' .
```

Running Docker
----------
Ensure that the variable `MODEL_DIR` in `run_text_generation_server.sh` is set to the location of the model directory. By default it is set to `MODEL_DIR=8b_chat_model_release`, which is the default name for the chat model. (For the base model, change this line to `MODEL_DIR=8b_base_model_release`.)

Running `sh docker_launch.sh` will start a model server that you can query via:

```
curl '<address of server>/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8' -d '{"prompts": ["human: Hello, how are you?\n\nadept:"], "tokens_to_generate": 128, "top_p": 0.9, "random_seed": 1234, "logprobs": false}'
```


Notes
-----

* The chat model is fine-tuned to expect inputs of the form: `human: {prompt}\n\nadept:`[^1]. To ensure best performance from this model, please use this format! You can see an example of this in the curl command above. To automatically wrap single-turn input prompts with this structure, you can modify the definition of `megatron/text_generation/api.py::generate_and_post_process` so that the default value for the argument `process_prompts_for_chat` is set to `True`. 
* We are releasing the model with tensor parallelism of 1.  In this configuration, the model requires an 80GB GPU to run naively.
It should be possible to fit the model on a 40GB card by removing the unused embeddings and reducing the maximum sequence length
(at the top of `run_text_generation_server.py`).
Quantization to 8-bit or lower would make also it fit with plenty of room to spare.
* We included the `.vocab` file so you can browse the vocabulary in plain text - this file is otherwise unused.

[^1]: Subsequent inputs should have the form `human: {prompt}\n\nadept: {output}\n\nhuman: {follow_up}\n\nadept:`
