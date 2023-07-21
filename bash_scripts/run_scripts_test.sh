for i in 1 2 3 4;
do
if [ $i == 1 ]; 
then
python3  -W ignore::UserWarning -m RNCapsGAN.test --name RNCapsGAN_VCC2SF3_VCC2TF1 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation/ --gpu_ids 0 --speaker_A_id VCC2SF3 --speaker_B_id VCC2TF1 --ckpt_dir results/New/RNCapsGAN_VCC2SF3_VCC2TF1/ckpts/ --load_epoch 500 --model_name generator_A2B
elif [ $i == 2 ];
then
python3  -W ignore::UserWarning -m RNCapsGAN.test --name RNCapsGAN_VCC2SF3_VCC2TM1 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation/ --gpu_ids 0 --speaker_A_id VCC2SF3 --speaker_B_id VCC2TM1 --ckpt_dir results/New/RNCapsGAN_VCC2SF3_VCC2TM1/ckpts/ --load_epoch 500 --model_name generator_A2B
elif [ $i == 3 ];
then
python3  -W ignore::UserWarning -m RNCapsGAN.test --name RNCapsGAN_VCC2SM3_VCC2TM1 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation/ --gpu_ids 0 --speaker_A_id VCC2SM3 --speaker_B_id VCC2TM1 --ckpt_dir results/New/RNCapsGAN_VCC2SM3_VCC2TM1/ckpts/ --load_epoch 500 --model_name generator_A2B
elif [ $i == 4 ];
then
python3  -W ignore::UserWarning -m RNCapsGAN.test --name RNCapsGAN_VCC2SM3_VCC2TF1 --save_dir results/New/ --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation/ --gpu_ids 0 --speaker_A_id VCC2SM3 --speaker_B_id VCC2TF1 --ckpt_dir results/New/RNCapsGAN_VCC2SM3_VCC2TF1/ckpts/ --load_epoch 500 --model_name generator_A2B
fi
done
