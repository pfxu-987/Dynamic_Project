nvidia-docker run --gpus 1  -it  --rm   -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=tt_baseline thufeifeibear/turbo_transformers_gpu:latest sh -c "cd /workspace/research && bash eval_throughput_seqs.sh"