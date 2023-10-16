#!/bin/bash
DATASET_DIR=/PATH/TO/EyeFairness
RESULT_DIR=.
MODEL_TYPE=( efficientnet ) # Options: efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

if [ ${MODALITY_TYPE} = 'oct_bscans_3d' ]; then
	LR=1e-4 
	BATCH_SIZE=6
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=1e-4
	BATCH_SIZE=10
else
	LR=1e-4
	BATCH_SIZE=6
fi

PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv

python ./scripts/train_glaucoma_fair.py \
		--data_dir ${DATASET_DIR}/AMD/ \
		--result_dir ${RESULT_DIR}/results/glaucoma_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}/${MODEL_TYPE}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE} \
		--model_type ${MODEL_TYPE} \
		--image_size 200 \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch-size ${BATCH_SIZE} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE}