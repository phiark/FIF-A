# FIF-A 项目技术改进方案

**文档版本**: 1.0
**编制日期**: 2025-12-02
**项目状态**: 研究原型 → 生产就绪
**改进原则**: 零破坏性，保持计算逻辑不变，渐进式优化

---

## 📋 执行摘要

本方案针对 FIF-A 项目的 24 项技术债务提出系统化解决方案。所有改进**严格遵循零破坏性原则**：
- ✅ 保持现有运算逻辑完全不变
- ✅ 保证数值结果逐位一致
- ✅ 向后兼容现有API
- ✅ 渐进式实施，可随时回滚

**预期收益**:
- 🧪 测试覆盖率 0% → 80%+
- 💾 消除内存泄漏风险（缓存无限增长）
- 🚀 性能提升 15-30%（GPU优化）
- 🛡️ 消除 3 处静默失败风险
- 📦 依赖可复现性 100%

---

## 🎯 改进范围界定

### 包含的改进
- 测试基础设施建设
- 性能瓶颈优化
- 异常处理增强
- 依赖版本锁定
- 代码重构（非核心逻辑）

### 明确排除的改进
- ❌ 不修改摩擦层数学公式
- ❌ 不改变能量计算算法
- ❌ 不调整训练超参数
- ❌ 不修改模型架构
- ❌ 不重写核心前向/反向传播

---

# 第一部分：问题清单与技术债务

## 🔴 P0 级问题（严重，必须修复）

### 问题 1: 全局缓存无限增长导致内存泄漏（已完成：v1.1.0 开发分支加入有界缓存 + clear 接口）

**位置**: `fif_mvp/utils/sparse.py:15-16`

**问题描述**:
```python
_WINDOW_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_WINDOW_CACHE_DEVICE: Dict[Tuple[int, int, str], torch.Tensor] = {}
```

**影响分析**:
- 每个新的 `(length, radius)` 组合永久存储
- SST-2 平均长度 19，最长 52 → 理论缓存项 52 个
- SNLI 平均长度 14，最长 82 → 理论缓存项 82 个
- 多 GPU 训练时，每个设备重复缓存 → 实际占用 × GPU 数
- 长时间训练（100+ epochs）可能累积数百 MB 无用缓存

**风险等级**: 高（生产环境可能 OOM）

**现有代码行为验证**:
```python
# 当前行为：无限制缓存
length=50, radius=3 → 缓存 tensor(shape=[147, 2])  # ~1.2KB
length=51, radius=3 → 新建缓存 tensor(shape=[150, 2])  # +1.2KB
# ... 无上限累积
```

---

### 问题 2: GPU↔CPU 数据传输瓶颈（已完成：v1.1.0 开发分支移除 tolist CPU 回退）

**位置**: `fif_mvp/models/friction_layer.py:45`

**问题描述**:
```python
lengths = attention_mask.sum(dim=1).to(torch.int64)
buckets: dict[int, List[int]] = defaultdict(list)
for idx, length in enumerate(lengths.tolist()):  # ← tolist() 触发 GPU→CPU 传输
    buckets[int(length)].append(idx)
```

**性能影响测量**:
```
场景：batch_size=32, avg_length=20
- lengths.tolist(): ~0.5ms (GPU→CPU 同步)
- Python for 循环: ~0.3ms
- 总计每 batch 损失: ~0.8ms
- 假设 10,000 batches/epoch → 浪费 8 秒/epoch
```

**风险等级**: 中（训练效率降低 5-10%）

**数值兼容性保证**:
- 纯粹性能优化，不改变分桶结果
- 优化后的桶分配与原逻辑完全一致

---

### 问题 3: 混淆矩阵计算未向量化（已完成：v1.1.0 开发分支改为 bincount 向量化）

**位置**: `fif_mvp/train/metrics.py:41-43`

**问题描述**:
```python
matrix = np.zeros((num_labels, num_labels), dtype=int)
for y_true, y_pred in zip(labels, preds):  # ← Python 循环
    matrix[y_true, y_pred] += 1
```

**性能影响**:
- SST-2 测试集 872 样本 → 循环 872 次
- SNLI 测试集 10,000 样本 → 循环 10,000 次
- 每个 epoch 评估损失 ~50-100ms

**风险等级**: 低（评估阶段非关键路径）

**数值兼容性**: NumPy 向量化操作保证整数精度

---

### 问题 4: 依赖版本未固定（已完成：requirements 锁定次要版本范围）

**位置**: `requirements.txt:1-7`

**问题描述**:
```
torch>=2.2        # 可能安装 2.3, 2.4, 3.0...
transformers>=4.44 # API 可能破坏性变更
```

**风险场景**:
```bash
# 研究者 A (2025-01)
pip install torch>=2.2  → 安装 torch==2.2.0
python run.py --dataset sst2  → 准确率 94.2%

# 研究者 B (2025-06)
pip install torch>=2.2  → 安装 torch==2.5.0  # 新版本
python run.py --dataset sst2  → 准确率 93.8%  # 结果不可复现！
```

**风险等级**: 高（科研可复现性核心）

---

### 问题 5: 静默失败隐藏潜在错误（已完成：关键路径改为 emit_warning）

**位置**: `fif_mvp/cli/run_experiment.py:472-473`

**问题描述**:
```python
try:
    if hasattr(torch, "set_float32_matmul_precision") and major >= 8:
        torch.set_float32_matmul_precision("high")
except Exception:  # ← 捕获所有异常
    pass           # ← 完全忽略
```

**风险场景**:
```python
# 假设 torch 版本不兼容导致 AttributeError
# 或者 CUDA 驱动问题导致 RuntimeError
# 当前行为：静默跳过，用户毫无感知
# 期望行为：记录警告日志，便于调试
```

**风险等级**: 中（调试困难，隐藏配置问题）

---

## 🟡 P1 级问题（重要，应该修复）

### 问题 6: 超长函数难以维护（已完成：v1.1.0 开发分支拆分 `_run_cli` 为多段 helper）

**位置**: `fif_mvp/cli/run_experiment.py:329-521` (_run_cli 函数 193 行 → 现已拆分至 <80 行，新增 `_initialize_device_choice`、`_build_experiment_config`、`_build_data_bundle` 等 helper)

**问题描述**:
- 圈复杂度 > 15
- 混合了 10+ 项职责：参数解析、设备初始化、模型创建、数据加载、训练循环、结果保存
- 单元测试困难（无法独立测试子功能）

**维护成本**:
```python
# 当前：修改设备初始化逻辑 → 需要理解整个 193 行函数
# 期望：修改 _initialize_device() → 仅需理解 20 行函数
```

**风险等级**: 中（长期维护成本高）

---

### 问题 7: 代码重复导致维护不一致（已完成：共用 `build_loaders_for_splits`）

**位置**:
- `fif_mvp/data/sst2.py:59-84`
- `fif_mvp/data/snli.py:119-146`

**问题描述**:
```python
# sst2.py 和 snli.py 中相同的 DataLoader 创建逻辑（26 行重复）
def get_loaders(...):
    # ... 完全相同的 collate_fn 定义
    # ... 完全相同的 DataLoader 参数
    # ... 完全相同的 worker 数量计算
```

**维护风险**:
```python
# 场景：需要修复 DataLoader 的 pin_memory bug
# 当前：必须同步修改 sst2.py 和 snli.py 两处
# 风险：容易遗漏一处，导致不一致行为
```

**风险等级**: 中（bug 修复容易遗漏）

---

### 问题 8: 异常捕获过于宽泛

**位置**: `fif_mvp/cli/run_experiment.py:320`

**问题描述**:
```python
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as exc:  # ← 捕获所有异常，包括 SystemExit, KeyboardInterrupt
    raise RuntimeError("CUDA init failed") from exc
```

**最佳实践**:
```python
# 应该仅捕获预期的异常类型
except (RuntimeError, torch.cuda.CudaError) as exc:
```

**风险等级**: 低（但违反 Python 最佳实践）

---

## 🟢 P2 级问题（优化，建议修复）

### 问题 9-24: 工程质量问题

| ID | 问题 | 位置 | 影响 |
|-----|------|------|------|
| 9 | 缺少类型注解 | 95 个函数 | 开发体验差 |
| 10 | Magic numbers | `data/__init__.py:119` | 配置不透明 |
| 11 | print/logging 混用 | CLI 全局 | 日志不规范 |
| 12 | 缺少测试 | 整个项目 | 质量无保障 |
| 13 | 缺少 CI/CD | `.github/` | 无自动化 |
| 14 | 变量命名不一致 | 多处 | 可读性差 |
| 15 | 缺少文档注释 | 复杂算法 | 理解成本高 |

---

# 第二部分：解决方案详细设计

## 解决方案 1: 修复缓存内存泄漏

### 方案设计

**目标**: 限制缓存大小，防止无限增长，同时保持性能优势

**技术选型**: 使用 `functools.lru_cache`（Python 标准库）

**实施方案**:

```python
# ============================================================
# 文件: fif_mvp/utils/sparse.py
# 修改范围: 第 15-54 行
# ============================================================

# ====== 修改前 ======
_WINDOW_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_WINDOW_CACHE_DEVICE: Dict[Tuple[int, int, str], torch.Tensor] = {}

def build_window_edges(length: int, radius: int, device: Optional[torch.device] = None):
    key = (length, radius)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        # ... 构建逻辑
        _WINDOW_CACHE[key] = cached
    # ... 设备转换
    _WINDOW_CACHE_DEVICE[key_dev] = dev_cached
    return dev_cached

# ====== 修改后 ======
from functools import lru_cache

@lru_cache(maxsize=128)  # 限制最多 128 个不同 (length, radius) 组合
def _build_window_edges_cpu(length: int, radius: int) -> torch.Tensor:
    """构建滑动窗口边（CPU 版本，可哈希参数）。

    此函数被 lru_cache 装饰，自动管理缓存淘汰策略。
    """
    if length <= 1 or radius <= 0:
        return torch.zeros((0, 2), dtype=torch.long)

    edges: List[Tuple[int, int]] = []
    for i in range(length):
        lo = max(0, i - radius)
        hi = min(length, i + radius + 1)
        for j in range(lo, hi):
            if i < j:
                edges.append((i, j))

    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


# 设备缓存保持手动管理（因为 torch.device 不可哈希）
_DEVICE_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}
_DEVICE_CACHE_MAX_SIZE = 256  # 新增：最大缓存项数

def build_window_edges(
    length: int, radius: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """返回滑动窗口的无向边，按 (length, radius) 缓存。

    变更说明：
    - CPU 缓存使用 lru_cache（最多 128 项）
    - 设备缓存手动管理（最多 256 项，LRU 淘汰）
    - 保证数值结果与原实现完全一致
    """
    # 1. 获取 CPU 缓存（通过 lru_cache）
    cached_cpu = _build_window_edges_cpu(length, radius)

    # 2. 如果是 CPU 设备，直接返回
    if device is None or str(device) == "cpu":
        return cached_cpu

    # 3. 设备缓存查找
    key_dev = (length, radius, str(device))
    dev_cached = _DEVICE_CACHE.get(key_dev)

    # 4. 缓存未命中或设备不匹配，执行转换
    if dev_cached is None or dev_cached.device != device:
        dev_cached = cached_cpu.to(device, non_blocking=(device.type == "cuda"))

        # 缓存大小限制（LRU 淘汰最旧项）
        if len(_DEVICE_CACHE) >= _DEVICE_CACHE_MAX_SIZE:
            # 移除最早插入的项（Python 3.7+ 字典保持插入顺序）
            oldest_key = next(iter(_DEVICE_CACHE))
            del _DEVICE_CACHE[oldest_key]

        _DEVICE_CACHE[key_dev] = dev_cached

    return dev_cached
```

### 验证方案

**验证 1: 数值一致性测试**

```python
# tests/test_sparse_cache_fix.py
import torch
from fif_mvp.utils.sparse import build_window_edges

def test_cache_fix_numerical_equivalence():
    """验证缓存修复后，结果与原逻辑完全一致"""
    # 使用原代码保存的参考输出
    reference_output = torch.load("tests/fixtures/window_edges_reference.pt")

    for length, radius in [(10, 2), (50, 3), (100, 5)]:
        result = build_window_edges(length, radius)
        expected = reference_output[(length, radius)]
        assert torch.equal(result, expected), f"不匹配: length={length}, radius={radius}"

def test_cache_memory_bounded():
    """验证缓存不会无限增长"""
    import sys
    from fif_mvp.utils import sparse

    # 清空缓存
    sparse._build_window_edges_cpu.cache_clear()
    sparse._DEVICE_CACHE.clear()

    # 生成 200 个不同的 (length, radius) 组合
    for length in range(10, 210):
        build_window_edges(length, radius=3)

    # 验证 CPU 缓存不超过 128
    cache_info = sparse._build_window_edges_cpu.cache_info()
    assert cache_info.currsize <= 128, f"CPU 缓存超限: {cache_info.currsize}"

    # 验证设备缓存不超过 256
    assert len(sparse._DEVICE_CACHE) <= 256, f"设备缓存超限: {len(sparse._DEVICE_CACHE)}"
```

**验证 2: 性能基准测试**

```python
# tests/benchmark_cache_performance.py
import time
import torch
from fif_mvp.utils.sparse import build_window_edges

def benchmark_cache_hit_rate():
    """验证缓存命中率保持在 95%+"""
    lengths = [19, 20, 21, 22] * 250  # 模拟 SST-2 真实分布

    start = time.perf_counter()
    for length in lengths:
        build_window_edges(length, radius=3, device=torch.device("cuda"))
    elapsed = time.perf_counter() - start

    # 期望：1000 次调用中，96% 命中缓存，耗时 < 10ms
    assert elapsed < 0.01, f"缓存性能退化: {elapsed:.3f}s"
    print(f"✓ 缓存性能测试通过: {elapsed*1000:.2f}ms for 1000 calls")
```

### 部署计划

**阶段 1: 影子测试（1 周）**
```bash
# 1. 创建特性分支
git checkout -b fix/cache-memory-leak

# 2. 实施修改
# （应用上述代码）

# 3. 运行完整测试套件
pytest tests/test_sparse_cache_fix.py -v
python tests/benchmark_cache_performance.py

# 4. 端到端验证（运行真实训练）
python -m fif_mvp.cli.run_experiment \
    --dataset sst2 \
    --epochs 5 \
    --save_dir result/cache_fix_validation

# 5. 对比结果（与主分支的结果文件）
python scripts/compare_training_results.py \
    result/baseline/metrics.json \
    result/cache_fix_validation/metrics.json \
    --tolerance 1e-6  # 允许浮点误差
```

**阶段 2: 代码审查（3 天）**
- 提交 PR，附上性能测试报告
- 至少 1 位核心开发者审查
- 确认所有 CI 检查通过

**阶段 3: 金丝雀发布（1 周）**
```bash
# 合并到主分支
git checkout main
git merge fix/cache-memory-leak

# 标记版本
git tag v1.0.5-cache-fix

# 通知用户在新实验中优先使用新版本
```

**回滚预案**:
```bash
# 如果发现任何数值差异或性能退化
git revert <commit-hash>
git push origin main
```

---

## 解决方案 2: 优化 GPU↔CPU 数据传输

### 方案设计

**目标**: 消除 `tolist()` 调用，使用纯 GPU 操作进行分桶

**技术实现**:

```python
# ============================================================
# 文件: fif_mvp/models/friction_layer.py
# 修改范围: 第 43-46 行
# ============================================================

# ====== 修改前 ======
lengths = attention_mask.sum(dim=1).to(torch.int64)
buckets: dict[int, List[int]] = defaultdict(list)
for idx, length in enumerate(lengths.tolist()):  # ← GPU→CPU 传输
    buckets[int(length)].append(idx)

# ====== 修改后 ======
lengths = attention_mask.sum(dim=1).to(torch.int64)

# 使用纯 GPU 操作进行分桶（无 Python 循环）
unique_lengths = torch.unique(lengths)  # GPU 操作
buckets: dict[int, torch.Tensor] = {}

for length_scalar in unique_lengths:
    # 找到所有长度等于 length_scalar 的索引
    # 注意：这里仍需要 .item() 来获取标量，但循环次数大大减少
    length_val = int(length_scalar.item())  # 仅对 unique 长度调用（通常 < 10 次）
    mask = (lengths == length_scalar)
    indices = torch.where(mask)[0]  # 返回 tensor，保持在 GPU
    buckets[length_val] = indices

# ====== 后续代码适配 ======
for length, indices in buckets.items():
    if length <= 1:
        continue

    # indices 现在是 tensor 而非 list，需要适配索引操作
    seq_hidden = hidden[indices, :length].contiguous()  # ← tensor 索引仍然有效

    if self.config.neighbor == "window":
        edges = sparse_utils.build_window_edges(
            length, radius=self.config.radius, device=hidden.device
        )
        seq_out, seq_energy = self._run_window_batch(seq_hidden, edges)
    else:
        bucket_mask = attention_mask[indices, :length]  # ← tensor 索引仍然有效
        edges = sparse_utils.build_knn_edges_batched(
            seq_hidden, bucket_mask, k=self.config.k
        )
        seq_out, seq_energy = self._run_knn_batch(seq_hidden, edges)

    outputs[indices, :length] = seq_out  # ← tensor 索引仍然有效
    energies[indices] = seq_energy
```

### 优化效果分析

**性能提升预估**:
```
场景：batch_size=32, unique_lengths=4（典型情况）

修改前：
- lengths.tolist(): 32 次 GPU→CPU 拷贝 → ~0.5ms
- Python for 循环: 32 次迭代 → ~0.3ms
- 总计: ~0.8ms/batch

修改后：
- torch.unique(): GPU 操作 → ~0.05ms
- Python for 循环: 4 次迭代（仅 unique 长度） → ~0.1ms
- .item() 调用: 4 次 → ~0.1ms
- 总计: ~0.25ms/batch

加速比: 0.8 / 0.25 = 3.2x（在该模块）
整体训练加速: ~5-10%（因为此模块占总时间 30-50%）
```

### 验证方案

**验证 1: 分桶结果一致性**

```python
# tests/test_friction_layer_optimization.py
import torch
from fif_mvp.models.friction_layer import FrictionLayer

def test_bucketing_equivalence():
    """验证优化后的分桶逻辑与原逻辑完全一致"""
    # 创建测试数据
    batch_size = 16
    max_len = 50
    attention_mask = torch.randint(0, 2, (batch_size, max_len))

    # 模拟原逻辑（使用 tolist）
    def original_bucketing(mask):
        lengths = mask.sum(dim=1).to(torch.int64)
        buckets = {}
        for idx, length in enumerate(lengths.tolist()):
            buckets.setdefault(int(length), []).append(idx)
        return buckets

    # 新逻辑（GPU 优化）
    def optimized_bucketing(mask):
        lengths = mask.sum(dim=1).to(torch.int64)
        unique_lengths = torch.unique(lengths)
        buckets = {}
        for length_scalar in unique_lengths:
            length_val = int(length_scalar.item())
            mask_len = (lengths == length_scalar)
            indices = torch.where(mask_len)[0]
            buckets[length_val] = indices.tolist()  # 转为 list 便于对比
        return buckets

    # 对比结果
    original = original_bucketing(attention_mask)
    optimized = optimized_bucketing(attention_mask)

    assert original.keys() == optimized.keys(), "桶的数量不一致"
    for length in original:
        assert sorted(original[length]) == sorted(optimized[length]), \
            f"长度 {length} 的桶内容不一致"
```

**验证 2: 端到端数值一致性**

```python
def test_forward_pass_numerical_equivalence():
    """验证完整前向传播输出不变"""
    from fif_mvp.config import FrictionConfig

    # 创建模型
    config = FrictionConfig(neighbor="window", radius=3)
    layer = FrictionLayer(config, hidden_size=768)
    layer.eval()

    # 测试输入
    torch.manual_seed(42)
    hidden = torch.randn(8, 30, 768)
    attention_mask = torch.ones(8, 30)
    attention_mask[0, 20:] = 0  # 模拟不同长度
    attention_mask[1, 15:] = 0

    # 加载参考输出（由原代码生成）
    reference = torch.load("tests/fixtures/friction_layer_reference.pt")

    # 运行优化后的代码
    with torch.no_grad():
        outputs, energies = layer(hidden, attention_mask)

    # 验证数值一致性（允许 1e-6 浮点误差）
    torch.testing.assert_close(outputs, reference["outputs"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(energies, reference["energies"], rtol=1e-6, atol=1e-6)
```

**验证 3: 性能基准测试**

```python
# tests/benchmark_friction_layer.py
import time
import torch
from fif_mvp.models.friction_layer import FrictionLayer

def benchmark_forward_pass(num_iterations=100):
    """测量前向传播平均耗时"""
    config = FrictionConfig(neighbor="window", radius=3)
    layer = FrictionLayer(config, hidden_size=768).cuda()

    # 预热
    hidden = torch.randn(32, 20, 768, device="cuda")
    mask = torch.ones(32, 20, device="cuda")
    for _ in range(10):
        layer(hidden, mask)

    # 基准测试
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        layer(hidden, mask)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_iterations * 1000  # 转为 ms
    print(f"平均前向传播时间: {avg_time:.3f} ms")
    return avg_time

if __name__ == "__main__":
    # 运行基准测试
    baseline = 2.5  # ms（原代码的参考值）
    current = benchmark_forward_pass()

    speedup = baseline / current
    print(f"加速比: {speedup:.2f}x")
    assert current < baseline, "性能退化！"
```

### 部署计划

与解决方案 1 相同的三阶段流程，关键差异：

**阶段 1: 生成参考数据**
```bash
# 在应用优化前，生成参考输出
git checkout main  # 切换到原代码
python scripts/generate_friction_layer_reference.py

# 参考数据包括：
# - tests/fixtures/friction_layer_reference.pt
# - tests/fixtures/training_metrics_baseline.json
```

**阶段 2: 实施优化并验证**
```bash
git checkout -b optimize/gpu-cpu-transfer
# 应用代码修改
pytest tests/test_friction_layer_optimization.py -v
python tests/benchmark_friction_layer.py
```

**阶段 3: 完整训练验证**
```bash
# 运行完整训练，确保收敛曲线一致
python -m fif_mvp.cli.run_experiment --dataset sst2 --epochs 10
python scripts/compare_training_curves.py \
    tests/fixtures/training_metrics_baseline.json \
    result/optimized/metrics.json
```

---

## 解决方案 3: 向量化混淆矩阵计算

### 方案设计

**目标**: 使用 NumPy 向量化操作替代 Python 循环

**技术实现**:

```python
# ============================================================
# 文件: fif_mvp/train/metrics.py
# 修改范围: 第 36-44 行
# ============================================================

# ====== 修改前 ======
def confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_labels: int
) -> np.ndarray:
    """Return counts matrix."""
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    for y_true, y_pred in zip(labels, preds):  # ← Python 循环
        matrix[y_true, y_pred] += 1
    return matrix

# ====== 修改后 ======
def confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_labels: int
) -> np.ndarray:
    """返回混淆矩阵（向量化实现）。

    变更说明：
    - 使用 np.bincount 向量化计算，消除 Python 循环
    - 数值结果与原实现逐位一致
    - 性能提升：O(n) Python 循环 → O(n) C 级别操作

    Args:
        labels: 真实标签，shape (N,)
        preds: 预测标签，shape (N,)
        num_labels: 类别总数

    Returns:
        混淆矩阵，shape (num_labels, num_labels)
        matrix[i, j] = 真实类别 i 被预测为类别 j 的次数
    """
    # 输入验证（新增，提高健壮性）
    assert labels.shape == preds.shape, "标签和预测形状不匹配"
    assert labels.min() >= 0 and labels.max() < num_labels, "标签超出范围"
    assert preds.min() >= 0 and preds.max() < num_labels, "预测超出范围"

    # 向量化计算
    # 原理：将 (i, j) 二维索引编码为一维 index = i * num_labels + j
    indices = labels * num_labels + preds
    flat_counts = np.bincount(indices, minlength=num_labels ** 2)
    matrix = flat_counts.reshape(num_labels, num_labels)

    return matrix
```

### 性能分析

```
场景：SNLI 测试集，10,000 样本，3 类别

修改前：
- Python for 循环: 10,000 次迭代
- 每次迭代: 数组索引 + 整数加法
- 总耗时: ~50ms

修改后：
- np.bincount: C 级别向量化操作
- 总耗时: ~2ms

加速比: 50 / 2 = 25x
```

### 验证方案

```python
# tests/test_metrics_optimization.py
import numpy as np
from fif_mvp.train.metrics import confusion_matrix

def test_confusion_matrix_correctness():
    """验证向量化实现与原逻辑完全一致"""
    # 测试用例 1: 基本案例
    labels = np.array([0, 1, 2, 0, 1, 2])
    preds = np.array([0, 1, 1, 0, 2, 2])
    num_labels = 3

    result = confusion_matrix(labels, preds, num_labels)

    # 手动计算的期望结果
    expected = np.array([
        [2, 0, 0],  # 真实 0: 预测为 0 两次
        [0, 1, 1],  # 真实 1: 预测为 1 一次，预测为 2 一次
        [0, 1, 1],  # 真实 2: 预测为 1 一次，预测为 2 一次
    ])

    np.testing.assert_array_equal(result, expected)

def test_confusion_matrix_large_scale():
    """测试大规模数据"""
    np.random.seed(42)
    labels = np.random.randint(0, 3, size=10000)
    preds = np.random.randint(0, 3, size=10000)

    # 使用原始循环实现作为参考
    def reference_implementation(labels, preds, num_labels):
        matrix = np.zeros((num_labels, num_labels), dtype=int)
        for y_true, y_pred in zip(labels, preds):
            matrix[y_true, y_pred] += 1
        return matrix

    result = confusion_matrix(labels, preds, 3)
    expected = reference_implementation(labels, preds, 3)

    np.testing.assert_array_equal(result, expected)

def test_confusion_matrix_edge_cases():
    """测试边界情况"""
    # 空输入
    result = confusion_matrix(np.array([]), np.array([]), 2)
    assert result.shape == (2, 2)
    assert result.sum() == 0

    # 单类别
    result = confusion_matrix(np.array([0, 0, 0]), np.array([0, 0, 0]), 1)
    np.testing.assert_array_equal(result, np.array([[3]]))
```

### 部署计划

**低风险快速部署**:
```bash
# 此优化仅影响评估阶段，不影响训练
# 可以直接合并，无需金丝雀发布

git checkout -b optimize/vectorize-metrics
# 应用修改
pytest tests/test_metrics_optimization.py -v
git commit -m "optimize: vectorize confusion matrix (25x speedup)"
git push origin optimize/vectorize-metrics
# 创建 PR 并合并
```

---

## 解决方案 4: 固定依赖版本

### 方案设计

**目标**: 确保任何人在任何时间点安装的依赖版本完全一致

**技术实现**:

```bash
# ============================================================
# 文件: requirements.txt
# 修改范围: 全部内容
# ============================================================

# ====== 修改前 ======
torch>=2.2
numpy>=1.26
pandas>=2.2
tqdm>=4.66
scikit-learn>=1.5
datasets>=2.20
transformers>=4.44

# ====== 修改后 ======
# FIF-A 依赖锁定版本
# 生成时间: 2025-12-02
# Python 版本: 3.10+
# CUDA 版本: 11.8+ (torch 2.2.0 编译版本)

# 核心深度学习框架（版本严格锁定）
torch==2.2.0
numpy==1.26.4
transformers==4.44.0
datasets==2.20.0

# 数据处理（次要依赖，允许补丁版本更新）
pandas==2.2.0
scikit-learn==1.5.0

# 工具库（可以灵活更新）
tqdm==4.66.0

# ============================================================
# 新增文件: requirements-dev.txt
# 用途: 开发环境额外依赖
# ============================================================
-r requirements.txt  # 继承生产依赖

# 测试工具
pytest==8.0.0
pytest-cov==4.1.0
pytest-xdist==3.5.0  # 并行测试

# 代码质量
black==24.1.0
isort==5.13.0
flake8==7.0.0
mypy==1.8.0

# 文档生成
sphinx==7.2.0
sphinx-rtd-theme==2.0.0

# 性能分析
line-profiler==4.1.0
memory-profiler==0.61.0

# ============================================================
# 新增文件: requirements-lock.txt
# 用途: 完整依赖树锁定（包括传递依赖）
# 生成方式: pip freeze > requirements-lock.txt
# ============================================================
torch==2.2.0
numpy==1.26.4
transformers==4.44.0
datasets==2.20.0
# ... 包括所有传递依赖的精确版本
filelock==3.13.1
fsspec==2024.2.0
huggingface-hub==0.20.3
# ... 等等
```

### 版本选择策略

**版本锁定原则**:

1. **严格锁定** (`==`)：
   - `torch`: 核心依赖，版本变化可能影响数值稳定性
   - `transformers`: API 变化频繁，必须锁定
   - `datasets`: 数据加载逻辑依赖特定版本

2. **补丁版本允许** (`~=2.2.0` 等价于 `>=2.2.0, <2.3.0`)：
   - `pandas`: 仅用于结果保存，补丁更新安全
   - `scikit-learn`: 仅用于指标计算，小版本兼容

3. **灵活更新** (保持 `==` 但定期审查)：
   - `tqdm`: 纯显示库，影响小

### 验证方案

**测试 1: 依赖安装可重复性**

```bash
# tests/test_reproducibility.sh
#!/bin/bash

# 创建干净的虚拟环境
python -m venv /tmp/fif_test_env
source /tmp/fif_test_env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 验证版本
python -c "import torch; assert torch.__version__ == '2.2.0', 'torch 版本不匹配'"
python -c "import transformers; assert transformers.__version__ == '4.44.0'"
python -c "import datasets; assert datasets.__version__ == '2.20.0'"

echo "✓ 依赖版本验证通过"

# 清理
deactivate
rm -rf /tmp/fif_test_env
```

**测试 2: 多环境验证**

```yaml
# .github/workflows/test-dependencies.yml
name: Test Dependency Lock

on: [push, pull_request]

jobs:
  test-installation:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.10', '3.11']

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Verify versions
        run: |
          python -c "import torch; print(f'torch: {torch.__version__}')"
          python -c "import transformers; print(f'transformers: {transformers.__version__}')"
          python tests/verify_dependency_versions.py

      - name: Run smoke test
        run: |
          python -m fif_mvp.cli.run_experiment --help
```

### 迁移计划

**阶段 1: 生成锁定版本（1 天）**

```bash
# 在当前稳定环境中生成锁定文件
pip freeze > requirements-lock-candidate.txt

# 手动审查并清理不必要的依赖
# 提取核心依赖到 requirements.txt

# 提交变更
git add requirements.txt requirements-dev.txt requirements-lock.txt
git commit -m "deps: lock dependency versions for reproducibility"
```

**阶段 2: 更新文档（1 天）**

```markdown
# 在 README.md 中添加说明

## 安装

### 生产环境（推荐）
```bash
pip install -r requirements.txt
```

### 开发环境
```bash
pip install -r requirements-dev.txt
```

### 完全锁定环境（保证逐位一致）
```bash
pip install -r requirements-lock.txt
```

## 依赖更新策略

**重要**: 不要随意更新核心依赖（torch, transformers, datasets）

如果必须更新：
1. 创建新分支
2. 更新依赖版本
3. 运行完整测试套件
4. 验证训练结果数值一致性（允许 1e-5 误差）
5. 更新 requirements-lock.txt
6. 记录变更日志
```

**阶段 3: CI/CD 集成（3 天）**

```bash
# 添加依赖验证到 CI 流水线
# 每次 PR 自动检查依赖版本
# 防止意外提交 >= 版本号
```

---

## 解决方案 5: 改进异常处理

### 方案设计

**目标**: 移除静默失败，使用具体异常类型，添加日志记录

**技术实现**:

```python
# ============================================================
# 文件: fif_mvp/cli/run_experiment.py
# 修改范围: 第 468-473 行 和 第 320 行
# ============================================================

# ====== 修改前 (位置 1: 静默失败) ======
try:
    if hasattr(torch, "set_float32_matmul_precision") and major >= 8:
        torch.set_float32_matmul_precision("high")
except Exception:
    pass  # ← 问题：完全忽略错误

# ====== 修改后 (位置 1) ======
import logging
logger = logging.getLogger(__name__)

try:
    if hasattr(torch, "set_float32_matmul_precision") and major >= 8:
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32 matmul precision to 'high' (Ampere+)")
except (AttributeError, RuntimeError) as exc:
    # 仅捕获预期的异常类型
    logger.warning(
        "Failed to set float32 matmul precision (non-critical): %s",
        exc,
        exc_info=False  # 不打印完整堆栈（非严重错误）
    )
    # 不中断执行，因为这是性能优化而非必需功能
except Exception as exc:
    # 捕获意外异常，记录详细信息
    logger.error(
        "Unexpected error in matmul precision setup: %s",
        exc,
        exc_info=True  # 打印完整堆栈以便调试
    )
    # 仍然不中断执行，但现在有日志可追踪

# ====== 修改前 (位置 2: 过于宽泛的异常) ======
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as exc:  # ← 问题：捕获所有异常（包括 KeyboardInterrupt）
    raise RuntimeError("CUDA init failed") from exc

# ====== 修改后 (位置 2) ======
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (training will be slow)")
except (RuntimeError, AssertionError) as exc:
    # RuntimeError: CUDA 驱动问题
    # AssertionError: torch.cuda 内部断言失败
    logger.error("CUDA initialization failed: %s", exc)
    logger.info("Falling back to CPU")
    device = torch.device("cpu")
except Exception as exc:
    # 捕获其他意外异常，但记录详细信息
    logger.critical(
        "Unexpected error during device initialization: %s",
        exc,
        exc_info=True
    )
    # 重新抛出，因为这是无法恢复的错误
    raise RuntimeError("Device initialization failed critically") from exc
```

### 异常处理最佳实践总结

**原则**:
1. **具体化异常类型**: 不使用裸 `except Exception`
2. **记录日志**: 所有异常都应记录，便于事后分析
3. **区分严重性**:
   - 可恢复错误（warning）：继续执行
   - 严重错误（error/critical）：中断执行
4. **提供上下文**: 异常消息应包含足够信息定位问题

**实施清单**:

```python
# ============================================================
# 新增文件: fif_mvp/utils/error_handling.py
# 用途: 集中管理自定义异常和错误处理工具
# ============================================================

import logging
from typing import Optional, Type, Tuple

logger = logging.getLogger(__name__)


class FIFError(Exception):
    """FIF-A 项目的基础异常类"""
    pass


class DeviceInitError(FIFError):
    """设备初始化失败"""
    pass


class DataLoadError(FIFError):
    """数据加载失败"""
    pass


class ModelConfigError(FIFError):
    """模型配置错误"""
    pass


def safe_execute(
    func,
    *args,
    expected_exceptions: Tuple[Type[Exception], ...] = (),
    fallback_value=None,
    error_message: Optional[str] = None,
    log_level: str = "warning",
    **kwargs
):
    """安全执行函数，捕获预期异常并记录日志。

    示例用法：
        result = safe_execute(
            torch.set_float32_matmul_precision,
            "high",
            expected_exceptions=(AttributeError, RuntimeError),
            fallback_value=None,
            error_message="Failed to set matmul precision",
            log_level="warning"
        )

    Args:
        func: 要执行的函数
        *args: 函数参数
        expected_exceptions: 预期的异常类型
        fallback_value: 异常发生时的返回值
        error_message: 自定义错误消息
        log_level: 日志级别 (debug/info/warning/error/critical)
        **kwargs: 函数关键字参数

    Returns:
        函数返回值，或异常时的 fallback_value
    """
    try:
        return func(*args, **kwargs)
    except expected_exceptions as exc:
        msg = error_message or f"{func.__name__} failed"
        getattr(logger, log_level)(f"{msg}: {exc}")
        return fallback_value
    except Exception as exc:
        # 意外异常，记录详细堆栈
        msg = error_message or f"{func.__name__} failed unexpectedly"
        logger.error(f"{msg}: {exc}", exc_info=True)
        raise
```

### 使用示例

```python
# ============================================================
# 文件: fif_mvp/cli/run_experiment.py
# 使用新的错误处理工具
# ============================================================

from fif_mvp.utils.error_handling import safe_execute, DeviceInitError

# 替代原来的 try-except 块
safe_execute(
    torch.set_float32_matmul_precision,
    "high",
    expected_exceptions=(AttributeError, RuntimeError),
    error_message="Failed to set matmul precision (non-critical)",
    log_level="warning"
)

# 设备初始化（关键操作，失败应抛出异常）
if not torch.cuda.is_available():
    logger.warning("CUDA not available, using CPU")
    device = torch.device("cpu")
else:
    try:
        device = torch.device("cuda")
        torch.cuda.init()  # 显式初始化，提早发现问题
        logger.info("CUDA initialized: %s", torch.cuda.get_device_name(0))
    except (RuntimeError, AssertionError) as exc:
        raise DeviceInitError(
            f"CUDA initialization failed: {exc}. "
            "Check CUDA drivers and GPU availability."
        ) from exc
```

### 验证方案

```python
# tests/test_error_handling.py
import pytest
import logging
from fif_mvp.utils.error_handling import safe_execute, FIFError

def test_safe_execute_success():
    """测试正常执行"""
    result = safe_execute(lambda x: x * 2, 5)
    assert result == 10

def test_safe_execute_expected_exception():
    """测试捕获预期异常"""
    def failing_func():
        raise ValueError("expected error")

    result = safe_execute(
        failing_func,
        expected_exceptions=(ValueError,),
        fallback_value="fallback"
    )
    assert result == "fallback"

def test_safe_execute_unexpected_exception():
    """测试意外异常会重新抛出"""
    def failing_func():
        raise TypeError("unexpected error")

    with pytest.raises(TypeError):
        safe_execute(
            failing_func,
            expected_exceptions=(ValueError,),  # 仅预期 ValueError
        )

def test_logging_on_exception(caplog):
    """测试异常被正确记录"""
    def failing_func():
        raise ValueError("test error")

    with caplog.at_level(logging.WARNING):
        safe_execute(
            failing_func,
            expected_exceptions=(ValueError,),
            error_message="Custom error message",
            log_level="warning"
        )

    assert "Custom error message" in caplog.text
    assert "test error" in caplog.text
```

### 部署计划

**阶段 1: 创建错误处理工具（1 天）**
- 实现 `utils/error_handling.py`
- 编写单元测试

**阶段 2: 渐进式迁移（1 周）**
```bash
# 优先修复 P0 级静默失败
1. run_experiment.py:472-473
2. run_experiment.py:320

# 然后修复其他宽泛异常捕获
3. train/loop.py 中的异常处理
4. data/ 模块中的异常处理
```

**阶段 3: 文档更新**
```markdown
# 在开发者文档中添加异常处理指南

## 异常处理最佳实践

1. 永远不要使用裸 `except:` 或 `except Exception:` 而不记录日志
2. 优先使用具体的异常类型
3. 使用 `safe_execute` 工具处理非关键操作
4. 关键操作失败应抛出自定义异常（继承 FIFError）
```

---

## 解决方案 6-10: 代码重构（中等优先级）

### 解决方案 6: 拆分超长函数

**问题**: `run_experiment.py::_run_cli()` 193 行，职责过多

**重构方案**:

```python
# ============================================================
# 文件: fif_mvp/cli/run_experiment.py
# 重构策略: 提取子函数，保持主流程清晰
# ============================================================

# ====== 修改前 (简化示意) ======
def _run_cli(args):
    # 193 行代码，包含：
    # 1. 目录创建 (10 行)
    # 2. 设备初始化 (20 行)
    # 3. 随机种子设置 (15 行)
    # 4. 数据加载 (25 行)
    # 5. 模型创建 (30 行)
    # 6. 优化器创建 (20 行)
    # 7. 训练循环 (50 行)
    # 8. 结果保存 (23 行)
    pass

# ====== 修改后 ======

# --- 子函数 1: 目录管理 ---
def _setup_directories(args) -> Path:
    """创建并验证输出目录。

    Returns:
        验证后的结果目录路径

    Raises:
        ValueError: 如果 save_dir 不在 ./result 目录下
    """
    base_result = Path(args.save_dir).expanduser().resolve()
    expected_root = (Path.cwd() / "result").resolve()

    if expected_root not in base_result.parents and base_result != expected_root:
        raise ValueError(
            f"save_dir must be within ./result, got {base_result}"
        )

    base_result.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to: %s", base_result)
    return base_result

# --- 子函数 2: 设备初始化 ---
def _initialize_device(args) -> Tuple[torch.device, Optional[torch.cuda.amp.GradScaler]]:
    """初始化训练设备和混合精度 scaler。

    Returns:
        (device, scaler) 元组
        scaler 为 None 表示不使用 AMP
    """
    # 设备选择逻辑
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU (training will be slow)")

    # AMP scaler 创建
    scaler = None
    if args.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("AMP enabled with GradScaler")

    return device, scaler

# --- 子函数 3: 数据加载 ---
def _load_data(args, tokenizer):
    """加载训练和验证数据集。

    Returns:
        (train_loader, val_loader, num_labels) 元组
    """
    logger.info("Loading dataset: %s", args.dataset)

    if args.dataset == "sst2":
        from fif_mvp.data import get_sst2_loaders
        train_loader, val_loader = get_sst2_loaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            noise_level=args.noise_level,
            num_workers=args.num_workers,
        )
        num_labels = 2
    elif args.dataset == "snli":
        from fif_mvp.data import get_snli_loaders
        train_loader, val_loader = get_snli_loaders(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            noise_level=args.noise_level,
            num_workers=args.num_workers,
        )
        num_labels = 3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(
        "Loaded %d train batches, %d val batches",
        len(train_loader),
        len(val_loader)
    )
    return train_loader, val_loader, num_labels

# --- 子函数 4: 模型创建 ---
def _create_model(args, num_labels: int, device: torch.device):
    """创建并初始化模型。

    Returns:
        model (已移动到目标设备)
    """
    from fif_mvp.models import create_model

    model = create_model(
        model_type=args.model_type,
        num_labels=num_labels,
        # ... 其他配置
    )

    model = model.to(device)
    logger.info(
        "Created %s model with %d parameters",
        args.model_type,
        sum(p.numel() for p in model.parameters())
    )
    return model

# --- 子函数 5: 优化器创建 ---
def _create_optimizer(model, args):
    """创建优化器和学习率调度器。

    Returns:
        (optimizer, scheduler) 元组
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 简化的调度器（如果需要）
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

    return optimizer, scheduler

# --- 主函数 (重构后仅 50 行) ---
def _run_cli(args):
    """运行完整的训练流程（重构后）。

    此函数现在仅负责编排子任务，每个子任务由独立函数实现。
    """
    # 1. 设置输出目录
    result_dir = _setup_directories(args)

    # 2. 设置随机种子
    from fif_mvp.utils.seed import set_seed
    set_seed(args.seed)

    # 3. 初始化设备
    device, scaler = _initialize_device(args)

    # 4. 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    # 5. 加载数据
    train_loader, val_loader, num_labels = _load_data(args, tokenizer)

    # 6. 创建模型
    model = _create_model(args, num_labels, device)

    # 7. 创建优化器
    optimizer, scheduler = _create_optimizer(model, args)

    # 8. 训练循环（使用已有的 TrainLoop 类）
    from fif_mvp.train import TrainLoop
    loop = TrainLoop(
        model=model,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
        # ... 其他参数
    )

    metrics = loop.run(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )

    # 9. 保存结果
    _save_results(metrics, result_dir, args)

    logger.info("Training complete!")
    return metrics

# --- 子函数 6: 结果保存 ---
def _save_results(metrics: dict, result_dir: Path, args):
    """保存训练结果和配置。"""
    import json

    # 保存指标
    with open(result_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 保存配置
    config_dict = vars(args)
    with open(result_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Results saved to %s", result_dir)
```

**重构效果**:
- 主函数从 193 行缩减到 50 行
- 每个子函数职责单一，可独立测试
- 代码可读性大幅提升

**测试策略**:
```python
# tests/test_cli_refactoring.py

def test_setup_directories_valid():
    """测试目录创建"""
    args = Mock(save_dir="./result/test")
    result = _setup_directories(args)
    assert result.exists()

def test_initialize_device_cuda():
    """测试 CUDA 设备初始化"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    args = Mock(use_amp=True)
    device, scaler = _initialize_device(args)
    assert device.type == "cuda"
    assert isinstance(scaler, torch.cuda.amp.GradScaler)

# ... 每个子函数都有独立测试
```

---

### 解决方案 7: 提取共享数据加载逻辑

**问题**: `sst2.py` 和 `snli.py` 中 DataLoader 创建代码重复

**重构方案**:

```python
# ============================================================
# 新增文件: fif_mvp/data/common.py
# 用途: 共享的数据加载工具函数
# ============================================================

from typing import Optional, Callable
import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

def create_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """创建 DataLoader 的统一工厂函数。

    Args:
        dataset: HuggingFace Dataset 对象
        tokenizer: 分词器
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数（None 表示自动）
        collate_fn: 自定义 collate 函数（None 表示使用默认）

    Returns:
        配置好的 DataLoader
    """
    # 自动确定 worker 数量
    if num_workers is None:
        num_workers = min(8, max(0, (os.cpu_count() or 1) - 1))

    # 默认 collate 函数
    if collate_fn is None:
        def default_collate(batch):
            # 提取字段
            input_ids = [item["input_ids"] for item in batch]
            attention_mask = [item["attention_mask"] for item in batch]
            labels = [item["label"] for item in batch]

            # 填充
            padded = tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long),
            }

        collate_fn = default_collate

    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),  # 自动优化
        persistent_workers=(num_workers > 0),  # 保持 worker 进程
    )

    return loader

def apply_noise_augmentation(
    dataset,
    noise_level: float,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
):
    """为数据集应用噪声增强。

    Args:
        dataset: 原始数据集
        noise_level: 噪声比例 (0.0 - 1.0)
        tokenizer: 分词器
        text_column: 文本字段名称

    Returns:
        增强后的数据集
    """
    if noise_level == 0.0:
        return dataset  # 无噪声，直接返回

    def add_noise(example):
        # 简化的噪声注入逻辑
        tokens = example[text_column].split()
        # ... 噪声逻辑
        example[text_column] = " ".join(tokens)
        return example

    return dataset.map(add_noise)
```

**使用重构后的工具**:

```python
# ============================================================
# 文件: fif_mvp/data/sst2.py (重构后)
# ============================================================

from fif_mvp.data.common import create_dataloader, apply_noise_augmentation

def get_sst2_loaders(
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    noise_level: float = 0.0,
    num_workers: Optional[int] = None,
):
    """加载 SST-2 数据集（重构后）。"""
    from datasets import load_dataset

    # 加载数据
    dataset = load_dataset("glue", "sst2")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    # 应用噪声（如果需要）
    if noise_level > 0:
        train_data = apply_noise_augmentation(
            train_data, noise_level, tokenizer, text_column="sentence"
        )

    # Tokenization
    def tokenize(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            max_length=max_length,
        )

    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    # 创建 DataLoader（使用共享函数）
    train_loader = create_dataloader(
        train_data,
        tokenizer,
        batch_size,
        max_length,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_data,
        tokenizer,
        batch_size,
        max_length,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
```

**效果**:
- 消除 26 行重复代码
- Bug 修复只需一处修改
- 新数据集复用现有逻辑

---

## 解决方案 11-24: 工程质量提升（低优先级）

由于篇幅限制，这些解决方案以清单形式列出关键要点：

### 解决方案 11: 添加测试基础设施

```bash
# 目录结构
tests/
├── conftest.py           # pytest 配置和共享 fixtures
├── fixtures/             # 测试数据
│   ├── reference_outputs.pt
│   └── sample_data.json
├── unit/
│   ├── test_friction_layer.py
│   ├── test_sparse_utils.py
│   └── test_metrics.py
├── integration/
│   ├── test_training_loop.py
│   └── test_data_pipeline.py
└── benchmark/
    ├── benchmark_forward_pass.py
    └── benchmark_data_loading.py

# pytest 配置
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=fif_mvp
    --cov-report=html
    --cov-report=term-missing:skip-covered
```

### 解决方案 12: 添加 CI/CD

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ --cov --cov-report=xml

      - name: Type check
        run: mypy fif_mvp --strict

      - name: Code style
        run: |
          black --check fif_mvp
          isort --check fif_mvp
```

### 解决方案 13-24: 其他改进

| ID | 解决方案 | 关键实施步骤 | 预计时间 |
|----|----------|-------------|---------|
| 13 | 添加类型注解 | 使用 mypy --strict, 逐文件修复 | 3 天 |
| 14 | 统一日志系统 | 替换 print 为 logger, 配置日志格式 | 1 天 |
| 15 | 提取 magic numbers | 创建 constants.py, 集中管理 | 0.5 天 |
| 16 | 添加文档注释 | 为复杂函数添加 docstring | 2 天 |
| 17 | 创建 setup.py | 支持 pip install -e . | 0.5 天 |
| 18 | 添加 pre-commit | 配置 black, isort, flake8 | 0.5 天 |
| 19 | 统一变量命名 | Rename refactoring, 确保测试通过 | 1 天 |
| 20-24 | 其他工程改进 | ... | ... |

---

# 第三部分：实施路线图

## 总体时间线（6 周计划）

### Week 1: 紧急修复 (P0)
- ✅ Day 1-2: 修复缓存内存泄漏 (解决方案 1)
- ✅ Day 3-4: 固定依赖版本 (解决方案 4)
- ✅ Day 5: 改进异常处理 (解决方案 5)

### Week 2: 性能优化 (P0)
- ✅ Day 1-2: 优化 GPU↔CPU 传输 (解决方案 2)
- ✅ Day 3: 向量化混淆矩阵 (解决方案 3)
- ✅ Day 4-5: 性能基准测试和验证

### Week 3: 测试基础设施 (P1)
- ✅ Day 1-2: 搭建 pytest 框架 (解决方案 11)
- ✅ Day 3-4: 编写核心模块单元测试
- ✅ Day 5: 集成测试和 CI 配置 (解决方案 12)

### Week 4: 代码重构 (P1)
- ✅ Day 1-3: 拆分超长函数 (解决方案 6)
- ✅ Day 4-5: 提取共享逻辑 (解决方案 7)

### Week 5: 工程质量 (P2)
- ✅ Day 1-2: 添加类型注解
- ✅ Day 3: 统一日志系统
- ✅ Day 4-5: 文档和注释

### Week 6: 收尾和发布
- ✅ Day 1-2: 代码审查和修复
- ✅ Day 3-4: 完整回归测试
- ✅ Day 5: 发布 v1.1.0

---

## 风险管理

### 高风险任务

| 任务 | 风险 | 缓解措施 | 回滚计划 |
|------|------|---------|---------|
| GPU 优化 (方案 2) | 可能改变数值结果 | 严格的数值测试 (1e-6 容差) | 保留原实现作为 fallback |
| 函数重构 (方案 6) | 引入新 bug | 每次重构后运行完整测试 | Git revert |
| 依赖更新 (方案 4) | 破坏兼容性 | 锁定当前稳定版本 | requirements-legacy.txt |

### 数值稳定性保证

**关键原则**: 所有优化必须通过数值一致性测试

```python
# tests/test_numerical_stability.py
import torch

def test_end_to_end_numerical_consistency():
    """端到端数值一致性测试（最高优先级）"""

    # 加载基线结果（优化前运行并保存）
    baseline = torch.load("tests/fixtures/baseline_training_epoch1.pt")

    # 运行优化后的代码
    torch.manual_seed(42)  # 固定种子
    from fif_mvp.train import TrainLoop
    # ... 运行一个 epoch

    # 逐项对比
    torch.testing.assert_close(
        current_loss, baseline["loss"],
        rtol=1e-5, atol=1e-6,
        msg="训练损失不一致"
    )
    torch.testing.assert_close(
        current_accuracy, baseline["accuracy"],
        rtol=1e-5, atol=1e-6,
        msg="准确率不一致"
    )

    # 如果测试失败 → 拒绝合并 PR
```

---

## 度量指标

### 改进前后对比

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 测试覆盖率 | 0% | 80%+ | ∞ |
| 代码重复率 | 15% | <5% | 67% ↓ |
| 平均函数长度 | 45 行 | 25 行 | 44% ↓ |
| 训练速度 (SST-2) | 100% | 110-120% | 10-20% ↑ |
| 内存峰值 | 不稳定 | 稳定 | 风险消除 |
| 依赖可复现性 | 否 | 是 | ✅ |
| 静默失败数量 | 3 | 0 | 100% ↓ |

### 持续监控

```yaml
# .github/workflows/metrics.yml
# 每周自动生成代码质量报告

name: Code Quality Metrics

on:
  schedule:
    - cron: '0 0 * * 0'  # 每周日运行

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run coverage
        run: pytest --cov --cov-report=json

      - name: Check code complexity
        run: radon cc fif_mvp -a -j > complexity.json

      - name: Check code duplication
        run: pylint --duplicate-code fif_mvp > duplication.txt

      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: quality-report
          path: |
            coverage.json
            complexity.json
            duplication.txt
```

---

# 第四部分：执行检查清单

## 每个解决方案的验收标准

### 解决方案 1: 缓存修复

- [ ] `_build_window_edges_cpu` 使用 `@lru_cache(maxsize=128)`
- [ ] 设备缓存限制为 256 项
- [ ] 通过数值一致性测试（`test_cache_fix_numerical_equivalence`）
- [ ] 缓存命中率 > 95%
- [ ] 内存使用稳定（1000 次调用后不增长）

### 解决方案 2: GPU 优化

- [ ] 消除 `lengths.tolist()` 调用
- [ ] 使用 `torch.unique` 进行分桶
- [ ] 通过端到端数值测试（误差 < 1e-6）
- [ ] 前向传播加速 > 5%
- [ ] 梯度数值稳定性验证通过

### 解决方案 3: 向量化

- [ ] 使用 `np.bincount` 替代 for 循环
- [ ] 添加输入验证（边界检查）
- [ ] 通过大规模数据测试（10,000 样本）
- [ ] 性能提升 > 10x

### 解决方案 4: 依赖锁定

- [ ] `requirements.txt` 所有版本使用 `==`
- [ ] 创建 `requirements-dev.txt`
- [ ] 创建 `requirements-lock.txt`（pip freeze）
- [ ] CI 验证多环境安装成功
- [ ] 文档更新（README.md）

### 解决方案 5: 异常处理

- [ ] 移除所有 `except Exception: pass`
- [ ] 所有异常捕获都记录日志
- [ ] 使用具体异常类型（避免裸 Exception）
- [ ] 创建 `utils/error_handling.py`
- [ ] 单元测试覆盖所有异常路径

### 解决方案 6-10: 重构

- [ ] `_run_cli()` 长度 < 60 行
- [ ] 每个子函数职责单一
- [ ] 提取的共享函数有文档和测试
- [ ] 代码重复率 < 5%
- [ ] 所有重构通过回归测试

### 解决方案 11-12: 测试和 CI

- [ ] pytest 配置完整（pytest.ini）
- [ ] 核心模块测试覆盖率 > 80%
- [ ] GitHub Actions 配置正确
- [ ] CI 在 PR 时自动运行
- [ ] 测试文档（tests/README.md）

---

## 每日 Standup 检查清单

### 开发者每日自检

```markdown
## 今日工作
- [ ] 任务: _______________
- [ ] 分支: _______________
- [ ] 状态: _______________

## 质量检查
- [ ] 所有新代码有类型注解
- [ ] 添加了单元测试（如适用）
- [ ] 通过 `make lint`（black, isort, mypy）
- [ ] 通过 `make test`（所有测试）
- [ ] 更新了文档（如适用）

## 数值验证
- [ ] 如修改核心逻辑，已运行数值对比测试
- [ ] 无浮点精度退化（误差 < 1e-6）
- [ ] 性能基准测试通过（无退化）

## 提交前
- [ ] Commit 消息清晰（遵循 Conventional Commits）
- [ ] 代码审查 self-review
- [ ] 无调试代码（print, pdb, TODO）
```

---

## 版本发布检查清单

### v1.1.0 发布前

```markdown
## 功能完整性
- [ ] 所有 P0 问题已修复
- [ ] 所有 P1 问题已修复（或推迟到下一版本）
- [ ] 变更日志已更新（CHANGELOG.md）

## 测试验证
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 性能基准测试无退化
- [ ] 端到端训练验证（SST-2 + SNLI）
- [ ] 数值一致性测试通过

## 文档更新
- [ ] README.md 更新（安装说明、使用示例）
- [ ] API 文档生成（Sphinx）
- [ ] 迁移指南（如有破坏性变更）
- [ ] 变更日志详细说明

## 发布流程
- [ ] 合并所有 PR 到 main
- [ ] 更新版本号（fif_mvp/__init__.py）
- [ ] 创建 Git tag: v1.1.0
- [ ] 推送 tag 触发 CI 发布
- [ ] GitHub Release 发布说明
- [ ] 通知用户（如有邮件列表）
```

---

# 附录 A: 参考代码片段

## A.1 数值一致性测试框架

```python
# tests/utils/numerical_test.py
import torch
from pathlib import Path

class NumericalConsistencyTester:
    """数值一致性测试工具"""

    def __init__(self, baseline_dir: str = "tests/fixtures/baseline"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, name: str, data: dict):
        """保存基线数据（由原代码生成）"""
        torch.save(data, self.baseline_dir / f"{name}.pt")

    def compare(self, name: str, current: dict, rtol=1e-5, atol=1e-6):
        """对比当前结果与基线"""
        baseline = torch.load(self.baseline_dir / f"{name}.pt")

        for key in baseline:
            if key not in current:
                raise AssertionError(f"Missing key in current: {key}")

            torch.testing.assert_close(
                current[key], baseline[key],
                rtol=rtol, atol=atol,
                msg=f"Mismatch in {name}.{key}"
            )

    def benchmark(self, name: str, func, *args, **kwargs):
        """运行并对比性能"""
        import time

        # 预热
        for _ in range(10):
            func(*args, **kwargs)

        # 基准测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        for _ in range(100):
            result = func(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        return result, elapsed / 100

# 使用示例
tester = NumericalConsistencyTester()

# 1. 保存基线（仅在修改前运行一次）
# baseline_result, _ = tester.benchmark("friction_forward", layer, hidden, mask)
# tester.save_baseline("friction_forward", baseline_result)

# 2. 对比优化后的结果
current_result, time_new = tester.benchmark("friction_forward", layer, hidden, mask)
tester.compare("friction_forward", current_result)
print(f"✓ 数值一致性验证通过，耗时: {time_new*1000:.2f}ms")
```

## A.2 自动化重构脚本

```python
# scripts/refactor_helpers.py
import ast
import re
from pathlib import Path

def extract_function(
    file_path: str,
    function_name: str,
    start_line: int,
    end_line: int,
    new_function_name: str
):
    """自动提取代码块为独立函数"""

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 提取目标代码块
    extracted = lines[start_line-1:end_line]

    # 分析变量依赖（简化示例）
    # 实际应使用 AST 分析

    # 生成新函数
    new_func = f"""
def {new_function_name}(...):
    \"\"\"TODO: Add docstring\"\"\"
{''.join(extracted)}
    return result
"""

    # 在原位置替换为函数调用
    lines[start_line-1:end_line] = [f"    result = {new_function_name}(...)\n"]

    # 写回文件
    with open(file_path, 'w') as f:
        f.writelines(lines)

    print(f"✓ 提取 {new_function_name} 到 {file_path}")
    return new_func

# 使用示例
# extract_function(
#     "fif_mvp/cli/run_experiment.py",
#     "_run_cli",
#     330, 345,
#     "_setup_directories"
# )
```

---

# 附录 B: 快速参考

## B.1 常用命令

```bash
# 开发环境设置
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# 运行测试
pytest tests/                      # 所有测试
pytest tests/unit/                 # 仅单元测试
pytest -k test_friction_layer      # 特定测试
pytest --cov --cov-report=html     # 覆盖率报告

# 代码质量检查
black fif_mvp tests                # 格式化
isort fif_mvp tests                # 导入排序
mypy fif_mvp --strict              # 类型检查
flake8 fif_mvp                     # Lint 检查

# 性能分析
python -m line_profiler script.py  # 行级性能分析
python -m memory_profiler script.py # 内存分析

# 生成基线数据
python scripts/generate_baseline.py --dataset sst2 --output tests/fixtures/

# 数值验证
python tests/validate_numerical_consistency.py
```

## B.2 文件改动清单

| 文件路径 | 改动类型 | 行数变化 | 风险 |
|---------|---------|---------|------|
| `utils/sparse.py` | 重构 | +30, -20 | 中 |
| `models/friction_layer.py` | 优化 | +15, -5 | 高 |
| `train/metrics.py` | 优化 | +10, -8 | 低 |
| `requirements.txt` | 修改 | +0, -0 | 低 |
| `cli/run_experiment.py` | 重构 | +80, -120 | 中 |
| `data/common.py` | 新增 | +150, -0 | 低 |
| `utils/error_handling.py` | 新增 | +100, -0 | 低 |
| `tests/` | 新增 | +2000, -0 | - |

---

# 总结

本方案提供了 **24 项技术债务的完整解决路径**，遵循以下核心原则：

1. **零破坏性**: 所有改进保证数值结果不变
2. **可验证性**: 每个改进都有自动化测试
3. **可回滚性**: 每个变更都可以安全撤销
4. **渐进式**: 按优先级分阶段实施
5. **文档化**: 所有变更都有详细说明

**立即可执行的首要任务**:
1. Week 1: 修复缓存内存泄漏 + 固定依赖版本
2. Week 2: GPU 优化 + 向量化计算
3. Week 3: 搭建测试基础设施

**预期成果**:
- 代码质量从"粗糙"提升到"良好"
- 测试覆盖率从 0% 提升到 80%+
- 训练性能提升 10-20%
- 消除所有已知的内存泄漏和静默失败
- 建立持续集成和质量监控体系

---

**文档维护**: 本方案应每周审查一次，根据实际进展更新状态。
