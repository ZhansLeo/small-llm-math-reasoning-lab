# Small-LLM Mathematical Reasoning Lab

一个围绕 `Qwen/Qwen2.5-0.5B-Instruct` 和 GSM8K 构建的完整小参数 LLM
实验项目。重点不是追求单次高分，而是建立可复现的闭环：

```text
Baseline Evaluation
→ Error Analysis
→ Prompt Intervention
→ Random / Error-guided LoRA-SFT
→ Frozen Holdout Re-evaluation
→ Paired Statistical Analysis
```

## 最终结论

最终结果来自预先冻结的 GSM8K test IDs 101–300（200 题），采用 greedy
decoding 和统一 evaluator v4。

| Model condition | Prompt A | Prompt C | Strict format A/C |
|---|---:|---:|---:|
| Base | **47.0%** | 41.0% | 0% / 0% |
| Random SFT | 34.5% | 30.0% | 99% / 99% |
| Error-guided SFT | 35.5% | 31.5% | 99% / 100% |

主要发现：

- Random SFT 和 Error-guided SFT 都显著改善了最终答案格式，但都没有提高
  数学正确率。
- Error-guided Prompt A 只比 Random Prompt A 高 1 个百分点；配对 McNemar
  检验 `p=0.8679`，没有证据表明两者存在稳定差异。
- Structured Prompt C 在三个模型条件下都名义下降，但 A/C 差异均未达到
  统计显著。
- Base Prompt A 的 106 个失败中，Relation Modeling Error 占 57.5%，远高于
  Arithmetic Error 的 10.4%。

完整方法、置信区间、逐题转移与代表案例见
[`evaluation/results/final_holdout_report.md`](evaluation/results/final_holdout_report.md)。

## 项目结构

```text
.
├── evaluation/                 # Prompt、推理后端、evaluator、统计与结果
│   ├── data/gsm8k/             # 本地 GSM8K train/test
│   └── results/                # 原始回答、比较、审计和最终报告
├── training/                   # Random / Error-guided LoRA 数据与训练流程
│   ├── configs/                # 固定训练配置
│   ├── manifests/              # 样本选择和数据哈希
│   ├── mining/                 # Base failure mining 结果
│   └── outputs/                # 训练指标；大权重与 checkpoint 不进入 Git
├── research_agent/             # 独立的早期 agent 学习原型
├── miniGPT.py                  # 从零实现 Transformer 的早期学习代码
├── requirements.txt
└── RELEASE_MANIFEST.json       # v1 冻结文件哈希
```

## 快速验证

建议使用 Python 3.10。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行无需模型下载的快速测试：

```powershell
python evaluation/test_parser.py
python evaluation/test_prompts.py
python evaluation/test_analysis.py
python evaluation/test_runner.py
```

检查冻结文件：

```powershell
python freeze_release.py --verify
```

如果本机还保留最终 adapter，可进行包含权重的完整检查：

```powershell
python freeze_release.py --verify --require-local-model-artifacts
```

## 复现实验

### 1. Baseline 与 Prompt

```powershell
python evaluation/evaluate.py --num-samples 10 --prompt prompt_a --run-label smoke
python evaluation/evaluate.py --num-samples 100 --prompt prompt_a
python evaluation/evaluate.py --num-samples 100 --prompt prompt_c
```

### 2. Random LoRA/SFT

```powershell
python training/prepare_random_sft.py
python training/test_training_data.py
python training/train_random_lora.py --smoke
python training/train_random_lora.py
```

### 3. Failure-mined Error-guided LoRA/SFT

```powershell
python training/mine_base_failures.py
python training/prepare_error_guided_sft.py
python training/test_error_guided.py
python training/train_error_guided_lora.py --smoke
python training/train_error_guided_lora.py
```

`training/run_error_guided_pipeline.py` 可以断点恢复地串联完整流程，但冻结的
IDs 101–300 不应再用于调参或模型选择。

## DeepSeek API

DeepSeek 只用于参考评测和辅助错误标签。复制 `.env.example` 到一个专用、被
Git 忽略的 env 文件，或直接设置环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "your-key"
python evaluation/evaluate_deepseek.py --num-samples 10 --prompt prompt_a --run-label smoke
```

代码不会读取共享 `.env`；默认只兼容专用 `(2).env`，也可以通过
`DEEPSEEK_ENV_FILE` 指定其他文件。任何真实 key 都不应进入 Git。

## 冻结与大文件策略

本仓库提交：源码、配置、数据/选样 manifest、训练指标、全部评测 raw
responses、统计结果和报告。以下可重建文件不提交：

- 中间 checkpoint、optimizer/scheduler/RNG 状态；
- smoke adapters；
- adapter 权重和重复 tokenizer 文件。

本地最终 adapter 的 SHA-256 仍写入冻结清单和协议文件。若要发布权重，建议
单独使用 Hugging Face Hub 或 Git LFS，并保持清单中的哈希不变。

## 结果使用边界

- 最终 holdout 只有 200 题，小差异需要结合置信区间和配对检验解释。
- 错误 taxonomy 是两轮模型辅助标注并经过一致性抽查，仍需完整人工复核后
  才能作为人工标注数据发布。
- 当前结论只适用于这里固定的模型、Prompt、1,000 样本 LoRA 设置与 holdout。
- 后续实验必须使用新的 holdout；不得根据 IDs 101–300 的结果继续调参后再在
  同一集合上报告“最终”性能。

本项目使用 [MIT License](LICENSE)，便于作为学习、研究与作品集项目公开展示。
