set env=vton_p38c118

CALL conda activate 

ECHO Y|conda remove -n %env% --all

rmdir /s /q D:\Users\User\anaconda3\envs\%env%



