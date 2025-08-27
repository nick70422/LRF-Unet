base=580
for i in {1..5}
do
    nnUNet_plan_and_preprocess -t $((base + i)) --verify_dataset_integrity
done

python3 ../dc_call.py preprocessing finished!