set env=vton_p38c118

cd /d "%~dp0"
cd ./../
CALL conda activate base
CALL conda activate %env%
python 005_test.py
pause