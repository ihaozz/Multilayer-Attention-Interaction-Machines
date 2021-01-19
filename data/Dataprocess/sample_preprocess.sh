cd data
mkdir Criteo
python3 ./Dataprocess/Criteo/preprocess.py
python3 ./Dataprocess/Kfold_split/stratifiedKfold.py
python3 ./Dataprocess/Criteo/scale.py
