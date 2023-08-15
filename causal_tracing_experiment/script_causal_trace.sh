# script for performing causal tracing experiment

# preparing data for causal tracing
# input should be the output file generated after model inference
python3 prepare_trace_data.py -i <out_json_file_after_model_inference> -svo <svo_annotated_file> -o <out_file>



# Running the causal tracing experiment
python3 causal_trace.py -i <out_file> --model_name <gpt2-large | gpt2-xl> --dataset <dataset_name>  --experiment <subject | object | verb> \
--checkpoint <model_checkpoint>
