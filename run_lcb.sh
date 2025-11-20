#!/bin/bash
# model: openreasoning_16k_step600
# -start_date 2024-10-01 --end_date 2025-02-28
model=sft-v5-base_1000-step-400_128k
vllm_port=34885
while true; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:${vllm_port}/health | grep -q "200"; then
        echo "Health check passed! Running command..."
        # 여기에 실행할 명령어 입력
    python -m lcb_runner.runner.main --model gpt-4-0613 --scenario codegeneration --evaluate --start_date 2024-07-01 --end_date 2025-01-01 --temperature 0.6 --multiprocess 64 --n 2 2>&1 | tee -a ${model}_LCB.log
        break
    else
        echo "Waiting for health check..."
        sleep 10
    fi
done
