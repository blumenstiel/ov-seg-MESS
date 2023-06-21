# run script with
# bash mess/setup_env.sh

# Create new environment "ovseg"
conda create --name ovseg -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovseg

# Install OVSeg requirements
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install -r requirements.txt
cd third_party/CLIP && python -m pip install -Ue .

# Install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas