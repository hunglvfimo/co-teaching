# Calculte dataset stats
	python dataset_stats.py --dataset MedEval17 --input_size 112
# Training
	python main.py --train --backbone ResNet50 --dataset G_FLOOD --input_size 224 --batch_size 16 --augment --loss_fn co_teaching --keep_rate 0.7 --num_gradual 10 --lr 0.001 --n_epoch 200 --epoch_decay_start 80  --eval_freq 10 --save_freq 10
# Evaluation
	python main.py --test --dataset MedEval17 --loss_fn co_teaching --input_size 112 --batch_size 8 --model1_name G_FLOOD_co_teaching_0.70_1_80.pth --model1_numclasses 2 --model2_name G_FLOOD_co_teaching_0.70_2_80.pth --model2_numclasses 2
# Fine tuning
	python main.py --train --backbone ResNet50 --dataset G_FLOOD --input_size 224 --batch_size 16 --augment --loss_fn co_teaching --keep_rate 0.7 --num_gradual 10 --lr 0.001 --n_epoch 200 --epoch_decay_start 80  --eval_freq 10 --save_freq 10 --model1_name resnet50_places365.pth.tar --model1_numclasses 365 --model2_name resnet50_places365.pth.tar --model2_numclasses 365