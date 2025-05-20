#!/bin/bash

echo "starting training"
cd ../../

BS=64

echo "starting training"
echo "3D"
python main.py --method 3D --result_path_base ./results/ --batch_size $BS
echo "2D"
python main.py --method 2D --result_path_base ./results/ --batch_size $BS
echo "2.5D"
python main.py --method 2.5D --result_path_base ./results/ --batch_size $BS
echo "2.75D"
python main.py --method 2.75D --result_path_base ./results/ --batch_size $BS
echo "2.75D_3channel"
python main.py --method 2.75D_3channel --result_path_base ./results/ --batch_size $BS

echo "2D_TL"
python main.py --method 2D_TL --result_path_base ./results/ --batch_size $BS
echo "2.5D_TL"
python main.py --method 2.5D_TL --result_path_base ./results/ --batch_size $BS
echo "2.75D_TL"
python main.py --method 2.75D_TL --result_path_base ./results/ --batch_size $BS
echo "2.75D_3channel_TL"
python main.py --method 2.75D_3channel_TL --result_path_base /home./results/ --batch_size $BS