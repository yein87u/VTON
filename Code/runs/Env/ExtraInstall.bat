set env=vton_p38c118

CALL conda activate 
CALL conda activate %env%


ECHO Y|pip install pytorch-fid




pause