set env=vton_p38c118

cd /d "%~dp0"
cd ./../
CALL conda activate 
CALL conda activate  %env%
python 000_ShowModel.py
PAUSE