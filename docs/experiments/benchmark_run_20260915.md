# Qwen / Llama L3 benchmark — 2026-09-15

本次使用 policy v2 重新评测已有候选。评测单位是一个 decoder layer 的 forward + backward（外部 cotangent），不包含 optimizer，也不是整模训练测速。

## 设置

| 项目 | 值 |
| --- | --- |
| GPU | 单卡 NVIDIA GH200 120GB；可见显存约 95 GiB |
| 软件 | PyTorch 2.11.0+cu128、Triton 3.6.0、Transformers 5.16.1 |
| 公共设置 | BF16，batch 2，sequence 2048，SDPA，关闭 KV cache 和 gradient checkpointing |
| Qwen3-0.6B | 第 14 层，H=1024，I=3072，HQ/HK=16/8，D=128 |
| Llama-3.2-1B | 第 8 层，H=2048，I=8192，HQ/HK=32/8，D=64 |
| 输入 | 从全模型执行捕获的权重、输入和上游梯度 |
| 重复 | 三轮独立 provider 进程，顺序 seed 11/22/33；每轮 warmup 3、10 samples × 3 timing blocks |
| 策略 | 每模型一份冻结的 evograd-t3-block-policy/2，三轮复用 |

`native` 是未修改的 HF decoder layer，通过 PyTorch autograd 执行 backward。`compile_all` 是各 L2 site 分别编译后的组合，不是整个 block 一次性编译。表中加速比按每轮对应 baseline 计算；失败项不计时。

单层只融合 post-attention residual + RMSNorm，保留初始 norm 和末尾 residual add，不覆盖跨层 residual fusion。

## 正确性

两模型各六个 patch set 的校准 controls 全部有效。Structural forward 逐位一致；backward 按独立 native envelope 判断。Preflight、purity、局部输出/梯度、调用次数和完整 block 检查均开启。

| 模型 | 三轮均通过的候选 | 三轮均拒绝的单 site |
| --- | --- | --- |
| Qwen | QKV、MLP、residual、三项组合、四项组合 | Attention：输出 rel L2 1.52e-3 > 2e-5 |
| Llama | QKV、MLP、三项组合、四项组合 | Attention：输出 rel L2 1.70e-3 > 2e-5；residual：gate_proj 梯度 rel L2 5.83e-3 > 5.23e-3 |

三项组合为 QKV + MLP + residual。组合 policy 使用该组合的可信 compile 偏差标定，阈值与单 site 不同；组合通过不能替代单 site 的正确性结论。两个模型的 exact-forward / backward ×10 负例均被 block gate 拒绝，未进入测速。

## Qwen3-0.6B 性能

| Provider | 三轮状态 | 三轮 fwd+bwd（ms） | vs native 范围 | vs site compile 范围 | 峰值显存（GiB） | saved state（MiB） |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 通过 | 3.657 / 3.681 / 3.554 | 1.00–1.00× | — | 0.45 | 281.7 |
| Compile all sites | 通过 | 2.847 / 2.728 / 2.923 | 1.22–1.35× | — | 0.51 | 193.7 |
| Compile attention | 通过 | 3.784 / 3.714 / 4.037 | 0.88–0.99× | — | 0.56 | 281.7 |
| Compile qkv_norm_rope | 通过 | 2.901 / 2.735 / 2.671 | 1.26–1.35× | — | 0.52 | 233.7 |
| Compile residual_rmsnorm | 通过 | 3.579 / 3.637 / 3.554 | 1.00–1.02× | — | 0.48 | 265.7 |
| Compile swiglu_mlp | 通过 | 3.670 / 3.804 / 3.673 | 0.97–1.00× | — | 0.46 | 257.7 |
| All four candidates | 通过 | 25.398 / 25.487 / 25.507 | 0.14–0.14× | 0.11–0.11× | 0.59 | 193.7 |
| Attention candidate | 拒绝 | — | — | — | — | — |
| QKV + MLP + residual | 通过 | 2.483 / 2.488 / 2.516 | 1.41–1.48× | 1.10–1.16× | 0.59 | 193.7 |
| QKV candidate | 通过 | 2.442 / 2.656 / 2.694 | 1.32–1.50× | 0.99–1.19× | 0.52 | 233.7 |
| Residual candidate | 通过 | 3.482 / 3.536 / 3.675 | 0.97–1.05× | 0.97–1.03× | 0.48 | 265.7 |
| MLP candidate | 通过 | 3.825 / 3.807 / 3.788 | 0.94–0.97× | 0.96–1.00× | 0.54 | 257.7 |

## Llama-3.2-1B 性能

| Provider | 三轮状态 | 三轮 fwd+bwd（ms） | vs native 范围 | vs site compile 范围 | 峰值显存（GiB） | saved state（MiB） |
| --- | --- | --- | --- | --- | --- | --- |
| Native | 通过 | 4.839 / 4.608 / 4.765 | 1.00–1.00× | — | 0.81 | 425.0 |
| Compile all sites | 通过 | 4.250 / 4.267 / 4.181 | 1.08–1.14× | — | 0.97 | 329.0 |
| Compile attention | 通过 | 4.917 / 4.872 / 4.918 | 0.95–0.98× | — | 0.95 | 425.0 |
| Compile qkv_rope | 通过 | 4.545 / 4.532 / 4.505 | 1.02–1.06× | — | 0.95 | 425.0 |
| Compile residual_rmsnorm | 通过 | 4.344 / 4.457 / 4.565 | 1.03–1.11× | — | 0.87 | 393.0 |
| Compile swiglu_mlp | 通过 | 4.743 / 4.617 / 4.608 | 1.00–1.03× | — | 0.86 | 361.0 |
| All four candidates | 通过 | 6.871 / 6.929 / 6.993 | 0.67–0.70× | 0.60–0.62× | 1.72 | 852.5 |
| Attention candidate | 拒绝 | — | — | — | — | — |
| QKV + MLP + residual | 通过 | 4.044 / 4.018 / 4.065 | 1.15–1.20× | 1.03–1.06× | 1.22 | 341.0 |
| QKV candidate | 通过 | 4.478 / 4.396 / 4.527 | 1.05–1.08× | 0.99–1.03× | 0.97 | 437.0 |
| Residual candidate | 拒绝 | — | — | — | — | — |
| MLP candidate | 通过 | 4.787 / 4.541 / 4.748 | 1.00–1.01× | 0.97–1.02× | 1.09 | 361.0 |
| Liger residual | 通过 | 4.459 / 4.698 / 4.616 | 0.98–1.09× | 0.95–0.99× | 0.87 | 393.0 |

## 结果与限制

- Qwen 三项组合相对 native 为 1.41–1.48×，相对 compile_all 为 1.10–1.16×。QKV 单 site 相对 compile 的收益不稳定；MLP 和 residual 单 site 没有稳定收益。
- Llama 三项组合相对 native 为 1.15–1.20×，相对 compile_all 为 1.03–1.06×。QKV 和 MLP 单 site 与各自 compile 基线接近。
- 含 attention candidate 的四项组合虽然通过组合 gate，但 Qwen 仅约 0.14× native，Llama 为 0.67–0.70× native。
- Saved state 减少不代表峰值显存一定下降，表中两项分别报告。
- 三轮来自同一节点和 allocation，未做跨节点重复、其它层或其它 batch/sequence。不能由这些结果推断整模训练收益。

## 工件标识

候选源码、captured tensors、policy 和逐轮原始结果保存在本地，不随仓库发布。以下标识用于区分本次评测工件；复现具体数值需要原始工件和匹配环境。

| 模型 / site | Candidate SHA256 前 12 位 |
| --- | --- |
| qwen3_0_6b / qkv_norm_rope | `40abfacaccd3` |
| qwen3_0_6b / attention | `e27c6128c346` |
| qwen3_0_6b / swiglu_mlp | `896f84ea7a18` |
| qwen3_0_6b / residual_rmsnorm | `a2fcdc137e51` |
| llama3_2_1b / qkv_rope | `c32041fbb8e1` |
| llama3_2_1b / attention | `dda5b0065854` |
| llama3_2_1b / swiglu_mlp | `b0e5414d6e4a` |
| llama3_2_1b / residual_rmsnorm | `537071c5e144` |

| 模型 | Case hash | Policy hash |
| --- | --- | --- |
| qwen3_0_6b | `cb1a0b8c9fdeba56e75c43ed19c43ef72b8b67be40d1ea09f531f67a3153f0a3` | `2f5a29eb4a9b9756f83c5612191458f6c20a17e724648363d419b34ebf87ed2c` |
| llama3_2_1b | `e7f5b9d4b8c32ab68b10e3bfb1f5ad7a76b7f92d22d2489f8214b0063d8f38eb` | `decdf78aa8e5ffc5db630c08264146299fd8d2d07dd9823f89de45c39e7aedb4` |

## 运行方式

准备本地 capture 和 candidate 后，可通过统一 CLI 运行。以下为单 site 示例，实际评测还包括组合和 compile providers：

```bash
evograd tier3-bench --scope block --model llama_3_2_1b \
  --block-source captured --artifact layer8.pt --layer-index 8 \
  --candidate qkv_rope=candidate.py --baseline none \
  --calibrate-with structural_identity,bound_pair_identity,trusted_torch_compile \
  --freeze-policy results/llama_block.policy.json \
  --order-seed 11 --warmup 3 --samples 10 --blocks 3 --noise-repeats 3 \
  --out results/llama_block.json
```

后续轮次使用同一 `--block-policy` 文件，改变 `--order-seed`。不同 case 或环境需要重新标定。检查规则见 [Block correctness](../L3_CORRECTNESS_CHECKS.md)。
