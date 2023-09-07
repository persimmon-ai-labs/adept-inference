sudo docker run \
    --rm \
    --gpus all \
    --device=/dev/infiniband \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env PYTHONPATH="." \
    -v $(pwd)/:/Adept_Inference/ \
    --network=host \
    --name=adept_inference \
    adeptdocker \
    bash -c "cd /Adept_Inference/; sh run_text_generation_server.sh";
