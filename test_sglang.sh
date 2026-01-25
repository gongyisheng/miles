export SGL_JIT_DEEPGEMM_PRECOMPILE=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=true
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=true
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=true
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT=true
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=false
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false

python -m sglang.launch_server \
	--model-path /root/Qwen2.5-0.5B-Instruct/ \
	--tokenizer-path /root/Qwen2.5-0.5B-Instruct/ \
	--trust-remote-code \
	--host localhost \
	--port 30000 \
	--nccl-port 30001 \
	--dist-init-addr localhost:30002 \
	--skip-server-warmup \
	--mem-fraction-static 0.7 \
	--chunked-prefill-size 2048 \
	--max-prefill-tokens 16384 \
	--schedule-policy fcfs \
	--random-seed 1234 \
	--watchdog-timeout 300 \
	--log-level info \
	--served-model-name /root/Qwen2.5-0.5B-Instruct/ \
	--attention-backend flashinfer \
	--sampling-backend pytorch \
	--grammar-backend xgrammar \
	--disable-radix-cache \
	--cuda-graph-max-bs 24 \
	--cuda-graph-bs 1 2 4 8 12 16 24 \
	--enable-memory-saver \
	--enable-deterministic-inference
