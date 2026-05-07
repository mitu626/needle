
MODEL_PATH="/root/paddlejob/wangyifei/.rapidfs/models/wangyifei/llama/Meta-Llama-3.1-8B-Instruct"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 
python -m needle \
  --model "$MODEL_PATH" \
  --port 8100 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.7
