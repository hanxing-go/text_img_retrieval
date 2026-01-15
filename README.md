# 文本→图像检索（模型+描述 JSON）— 空中/车辆细粒度演示

这是一个**完整的、可运行的**单机项目，用于：
- 当提供 `model` 时进行**精确匹配**（支持别名）
- 当缺少 `model` 或未匹配时，使用 CLIP 嵌入 + FAISS 进行**语义检索**（形状/纹理）
- 设计用于**<= 10k 图像**（但也可扩展），无需多租户，无需在线摄取。

> 默认：使用 **OpenCLIP** + **FAISS IndexFlatIP（精确 KNN）** 以实现最大精度。

---

## 0) 前置条件
- 推荐使用 Python 3.10+
- CUDA GPU 是可选的。CPU 也可以运行。

### 安装 PyTorch（GPU）
根据你的 CUDA 版本，按照官方选择器安装 PyTorch。
（如果你想使用 GPU，请在安装依赖项之前完成此步骤。）

然后：

```bash
pip install -r requirements.txt
```

---

## 1) 准备数据集

### 1.1 将图像放入文件夹
示例：
```
data/images/
  0001.jpg
  0002.png
  ...
```

### 1.2 创建元数据 JSONL
创建 `data/metadata.jsonl`，每行一个 JSON：

```json
{"image_id":"0001","filepath":"data/images/0001.jpg","model_std":"F-16C BLOCK 50","aliases":["F16","F-16","F-16C","F16C","BLK50","BLOCK50"],"type":"aircraft"}
{"image_id":"0002","filepath":"data/images/0002.jpg","model_std":"SU-27","aliases":["SU27","SU-27","FLANKER"],"type":"aircraft"}
```

注意：
- `model_std` 应该是你的规范化模型字符串（我们会在内部进行标准化）。
- `aliases` 应包括常见变体（带/不带连字符、空格等）。
- 你可以添加额外字段；它们将以 JSON 格式存储在数据库中。

---

## 2) 构建索引（离线，一次性）

```bash
python -m app.build_index \
  --meta data/metadata.jsonl \
  --out_dir data/index \
  --batch_size 64
```

输出：
- `data/index/index.faiss`
- `data/index/id_map.json`
- `data/index/meta.db`（SQLite）

---

## 3) 运行 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：
```bash
curl http://localhost:8000/health
```

---

## 4) 查询示例

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "model": "F-16C Block 50",
    "desc": "单发战斗机，后掠翼，单垂尾，无背景 侧视",
    "top_k": 20,
    "rerank": false
  }'
```

行为：
1) 如果 `model` 匹配已知别名 => 立即返回该模型的图像。
2) 否则，使用 CLIP 文本嵌入 + FAISS 检索最近的图像。

---

## 5) 性能预期（<=10k 图像）
- **模型匹配路径**：~2–10 毫秒
- **向量检索路径**（无重排序）：通常 ~3–20 毫秒（取决于 GPU/CPU）
- **1 秒以内**轻松实现；大多数查询应为**几十毫秒**。

---

## 6) 配置
编辑 `app/config.py` 或设置环境变量：
- `TIR_MODEL_NAME`（默认 `ViT-L-14`）
- `TIR_PRETRAINED`（默认 `openai`）
- `TIR_DEVICE`（`cuda` / `cpu`，默认自动）
- `TIR_INDEX_DIR`（默认 `data/index`）

---

## 7) 提高领域精度的注意事项
- 在 `aliases` 和 `model_std` 映射上投入精力——这可以保证“提供模型 => 精确结果”。
- 对于“无模型”情况，内置的提示集成有助于基于形状的检索。
- 如果你想进一步提高精度，可以添加**领域微调**（使用困难负样本进行 CLIP 对比微调）。

