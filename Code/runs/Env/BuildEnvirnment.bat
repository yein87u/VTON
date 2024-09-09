set env=vton_p38c118

CALL conda activate 
ECHO Y|conda create --name %env% python=3.8
CALL conda activate %env%

ECHO Y|pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

ECHO Y|pip install albumentations
ECHO Y|pip install scikit-learn

ECHO Y|pip install pytorch-model-summary
ECHO Y|pip install opencv-python
ECHO Y|pip install tensorboardX
ECHO Y|pip install tqdm
ECHO Y|pip install timm
ECHO Y|pip install einops
ECHO Y|pip install cmake
ECHO Y|pip install boost
ECHO Y|pip install dlib
ECHO Y|pip install fvcore
ECHO Y|pip install matplotlib 
ECHO Y|pip install cupy-cuda11x
ECHO Y|pip install pytorch-fid


PAUSE