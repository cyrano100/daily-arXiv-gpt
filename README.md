1. 在 `.env` 中配置 DashScope API 密钥

2. 命令行输入：
```bash
python get_recent_paper.py --keyword "graph" --save_dir ./data --max 10
```
- `keyword`: 要检索的关键词
- `save_dir`: 保存结果的目录
- `max`: 要检索的论文数量