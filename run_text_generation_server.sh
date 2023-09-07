# Copyright (c) 2023, ADEPT AI LABS INC.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
MODEL_DIR="8b_chat_model_release"
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $(hostname) --master_port 5003 run_text_generation_server.py --no-load-rng --no-load-optim --no-initialization --top_p 0.9 --port 6001 --micro-batch-size 1 --load ${MODEL_DIR} --use-flash-attn --sp-model-file ${MODEL_DIR}/adept_vocab.model
