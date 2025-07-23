mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
export PATH="~/anaconda3/bin:$PATH"
source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc

wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sh cuda_12.1.0_530.30.02_linux.run

# git clone --recurse-submodules https://github.com/noahwei682/mmsearch_r1.git
# cd mmsearch_r1
# git clone --recurse-submodules https://ghfast.top/https://github.com/noahwei682/mmsearch_r1.git
git clone --recurse-submodules https://github.com/noah888999666/multimodal-search-r1.git
cd multimodal-search-r1
# export HF_ENDPOINT=https://hf-mirror.com
# Init Conda Env
# conda config --remove-key channels
# conda config --add channels conda-forge
# conda config --add channels defaults

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# git clone --recurse-submodules https://github.com/noahwei682/multimodal-search-r1.git
# cd multimodal-search-r1
# Init Conda Env
conda create -n mmsearch_r1 python==3.10 -y
conda activate mmsearch_r1
# Install Dependencies
pip3 install -e ./verl
pip3 install vllm==0.8.2
pip3 install transformers==4.51.0
pip3 install flash-attn==2.7.4.post1
pip3 install scikit-learn==1.3.0
pip install sentence-transformers

# conda create -n mmsearch_r1 python==3.10 -y
# source activate mmsearch_r1
# conda activate mmsearch_r1
# # Install Dependencies
# pip3 install -e ./verl
# # pip install --upgrade --retries 10 --timeout 1000 torch==2.6.0
# # conda install -c pytorch pytorch=2.6.0 torchvision torchaudio
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# pip uninstall tensordict -y
# pip install tensordict==0.2.0  # or another compatible version
# pip3 install vllm==0.8.2
# pip3 install transformers==4.51.0
# # pip uninstall -y numpy && pip install numpy==1.24.3
# pip install "numpy>=1.22,<2.0.0"

# # stacked here
# # pip install flash-attn==2.4.3.post1
# # pip install flash-attn --use-pep517 --no-build-isolation
# # pip3 install flash-attn==2.7.4.post1
# pip uninstall flash-attn -y
# pip install flash-attn --no-build-isolation
# Init wandb
pip3 install wandb
export WANDB_API_KEY="12889579b4a78319f80e202e35156aa0f1edd9e4"
wandb login $WANDB_API_KEY

pip install huggingface_hub
#huggingface-cli logout
export HF_TOKEN=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
# huggingface-cli login --token $HF_TOKEN
git config --global credential.helper store
huggingface-cli login --add-to-git-credential
# pip install hydra-core
# pip install ray
# pip install numpy
# pip install pandas
# pip install tqdm


# 在run_mmsearch_r1_grpo.sh开头添加
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1

bash mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh

# git clone --recurse-submodules https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git
# cd multimodal-search-r1



# # Clone this repo with submodules
# git clone https://ghfast.top/https://github.com/EvolvingLMMs-Lab/multimodal-search-r1.git
# cd multimodal-search-r1
# # Init Conda Env
# conda create -n mmsearchr1 python==3.10 -y
# source activate mmsearchr1
# conda activate mmsearchr1
# # Install Dependencies
# pip3 install -e ./verl
# pip3 install vllm==0.8.2
# pip3 install transformers==4.51.0
# pip3 install flash-attn==2.7.4.post1
# # Init wandb
# pip3 install wandb
# export WANDB_API_KEY="XXX"
# wandb login $WANDB_API_KEY
