# 2026-11-19 实验记录

**后续更新**：SAR 直接查询后来补齐至 6/6 条记录；本目录保留发布当时 4/6 快照，修订统计见[11 月 20 日记录](../2026-11-20/sar-summary.json)。

- [Qwen 良性逐项记录](qwen-benign.json)：12 条完成原答、1 次启动后加载前资源停止、1 次加载前超时；数字标签组首项启动前拦截，后两项未执行。输入图及源记录 SHA-256 用于与本地归档核对，不表示原文件已公开。语义标签为既有人工判读，精确输出判据按条件记录。
- [SAR 脱敏汇总](sar-summary.json)：两种实验分母独立，来自共同干净正确的开发图。最大像素差使用 0–255 刻度，不是平均改变量。直接查询缺失不算失败；历史对照不算本日新增。未附私人样本、原始日志或攻击实施代码。
- [合成图生成与记录校验](../../../scripts/public_benign_records.py)和[测试](../../../tests/test_public_benign_records.py)：不启动模型、不访问网络、不读取私人输入。

## 良性协议

[脱敏协议索引](benign-protocol.json)按组列出计划分母、执行顺序、配对与停止口径；实际问题、精确判据和逐项终态见逐项记录。

Qwen 固定公开模型版本、离线确定性解码、上限 64 个新 token，逐项独立运行。计数配对固定用户问题，只改底栏；转写为独立调用，不证明计数时读取过文字。严格格式要求不追溯修改，允许短句的条件单列。数字标签准备 3/3 不是推理。辅助离线诊断完成一次（39.35 秒，13 个清单文件通过）；旧异常无图输入的 35 个 token 双向一致。未运行图片注入条件。

## 离线复核

从仓库根目录运行：

```text
python scripts/public_benign_records.py validate experiments/records/2026-11-19
python -m unittest discover -s tests -p test_public_benign_records.py -v
python scripts/public_benign_records.py generate public-squares.png
```

前两条只需 Python 标准库；生成使用 Pillow，拒绝覆盖文件。生成器仅覆盖本日无文字蓝色方块素材；不是完整 GPU 复现包。测试验证记录结构、分母及精确文本判据，不重判语义，不算新增模型实验。

发布前实测：记录校验通过；公开仓库单元测试 12/12 通过（其中本批新增 5 项）；在本地 Pillow 环境生成的 PNG 与保存的无文字方块图 SHA-256 一致。
