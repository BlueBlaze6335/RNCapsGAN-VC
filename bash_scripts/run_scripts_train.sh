for i in 1 2 3 4;
do
if [ $i == 1 ]; 
then
python3 -W ignore::UserWarning -m RNCapsGAN.train --name RNCapsGAN_VCC2SF3_VCC2TF1 --seed 0 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ --speaker_A_id VCC2SF3 --speaker_B_id VCC2TF1 --epochs_per_save 100 --epochs_per_plot 5 --num_epochs 502 --batch_size 1 --decay_after 1e4 --sample_rate 22050 --num_frames 64 --max_mask_len 25 --gpu_ids 0
elif [ $i == 2 ];
then
python3 -W ignore::UserWarning -m RNCapsGAN.train --name RNCapsGAN_VCC2SF3_VCC2TM1 --seed 0 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ --speaker_A_id VCC2SF3 --speaker_B_id VCC2TM1 --epochs_per_save 100 --epochs_per_plot 5 --num_epochs 502 --batch_size 1 --decay_after 1e4 --sample_rate 22050 --num_frames 64 --max_mask_len 25 --gpu_ids 0
elif [ $i == 3 ];
then
python3 -W ignore::UserWarning -m RNCapsGAN.train --name RNCapsGAN_VCC2SM3_VCC2TM1 --seed 0 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ --speaker_A_id VCC2SM3 --speaker_B_id VCC2TM1 --epochs_per_save 100 --epochs_per_plot 5 --num_epochs 502 --batch_size 1 --decay_after 1e4 --sample_rate 22050 --num_frames 64 --max_mask_len 25 --gpu_ids 0
elif [ $i == 4 ];
then
python3 -W ignore::UserWarning -m RNCapsGAN.train --name RNCapsGAN_VCC2SM3_VCC2TF1 --seed 0 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ --speaker_A_id VCC2SM3 --speaker_B_id VCC2TF1 --epochs_per_save 100 --epochs_per_plot 5 --num_epochs 502 --batch_size 1 --decay_after 1e4 --sample_rate 22050 --num_frames 64 --max_mask_len 25 --gpu_ids 0
fi
done
