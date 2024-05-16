for i in T5
do
    python3 main.py --predict_task 'label_diabetes' \
                    --token_type ${i} \
                    --models_type ${i} \
                    --batch_size 32 \
                    --epochs 50 \
                    --use_attention 'True' \
                    --early_prediction 'False'
done


