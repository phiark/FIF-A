# FIF-A Quick Start Guide

**文档元数据**
- 文档类型：实验启动速查
- 适用版本：v1.2.0+
- 硬件基线：Tesla V100 16G ×1
- 最后更新：2025-12-19
- 相关文档：`EXPERIMENT_LAUNCHER_GUIDE.md`, `README.md`, `scripts/experiments.yaml`

**快速导航**
- [最简运行](#1-最简运行)
- [快速选择器](#2-快速选择器)
- [结果查看](#3-结果查看)
- [进一步配置](#4-进一步配置)

---

## 1. 最简运行

```bash
./run.sh
```

运行 v1.2.0 标准实验（SNLI + SST-2，baseline + hybrid）。

---

## 2. 快速选择器

```bash
# 运行所有 SNLI 实验
./quick.sh snli

# 运行所有 SST-2 实验
./quick.sh sst2

# 运行单个实验
./quick.sh snli_baseline
./quick.sh sst2_hybrid

# 预览命令（不执行）
./quick.sh snli --dry-run
```

---

## 3. 结果查看

实验结果位于 `result/1_2_0/`：

```bash
# 查看测试集准确率
cat result/1_2_0/snli_baseline_*/test_summary.json | jq '.acc'

# 查看能量-错误相关性
cat result/1_2_0/snli_hybrid_*/energy_error_correlation.json | jq '.pearson_r'
```

---

## 4. 进一步配置

完整用法与 YAML 配置说明见 `EXPERIMENT_LAUNCHER_GUIDE.md`。
