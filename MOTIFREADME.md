임시방편으로 lcb_runner/runner/oai_runner.py 에 요청을 보내는 쪽에 localhost쪽으로 보내도록 하드코딩 해둬서 가능하게 해둠.

```bash
python -m lcb_runner.runner.main --model gpt-4-0613 --scenario codegeneration --evaluate --release_version release_v5 --continue_existing_with_eval --start_date 2024-07-01 --end_date 2025-01-01
```

위 명령어로 돌리면 됨.