# DD-Payne for LAMOST DR9

这是一个面向 LAMOST 低分辨率光谱的 DD-Payne 复现基础框架。它实现了：

- LAMOST DR9 `fits.gz` 读取、质量掩码、静止系校正、伪连续谱归一化和公共波长网格重采样；
- LAMOST 与 APOGEE/GALAH 等高分辨率标签表的天球坐标交叉匹配；
- 从 LAMOST 官方目录先生成候选 `obsid`，优先 Gaia ID、再用坐标回退匹配；
- 按恒星分组的 train/validation/test 划分，避免重复观测泄漏；
- 论文式逐像素两隐层 Payne 网络；
- 观测光谱重建损失与物理梯度谱正则项；
- 检查点、验证指标、逐星标签拟合和不确定度估计接口；
- 可在 CPU 上运行的小型端到端 smoke test。

> 重要：`spectra/` 中的 125,075 条光谱没有元素丰度监督标签，也没有 Kurucz 梯度谱。仅凭这些文件不能完成科学意义上的 DD-Payne 训练。需要先获得 APOGEE/GALAH 等标签并制作物理梯度库。

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip

# NVIDIA RTX 50 系显卡：先安装已在本机验证可用的 CUDA 13.0 构建
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.0+cu130"

.venv\Scripts\python -m pip install -e ".[dev]"

# 验证安装、模型、损失和训练循环
.venv\Scripts\ddpayne smoke-test --work-dir outputs/smoke

# 只扫描少量文件，检查本地 DR9 数据
.venv\Scripts\ddpayne inspect-data --spectra-root spectra/dr9-v2.0-lrs-fits --limit 100

# 构建完整元数据表
.venv\Scripts\ddpayne build-manifest --spectra-root spectra/dr9-v2.0-lrs-fits --output data/metadata/spectra.csv

# 目录级交叉匹配，输出候选 obsid 和训练标签（可直接读取 APOGEE FITS）
.venv\Scripts\ddpayne build-candidates --lamost-catalog data/metadata/spectra.csv `
  --labels data/external/allStar-dr17-synspec_rev1.fits `
  --output data/interim/lamost_apogee_candidates.csv `
  --quality apogee
```

当前本地光谱应使用 `data/metadata/spectra.csv`。文档中的
`data/external/lamost_full_lrs_catalog.fits` 或
`data/external/lamost_dr9_lrs_general.fits` 是官方全量 LAMOST 目录的占位路径，只有在你下载并保存该目录后才能作为 `--lamost-catalog` 使用。

若只使用 CPU，可跳过 CUDA 版 PyTorch 命令。其他显卡/驱动请以 PyTorch 官方安装选择器给出的命令为准。

完整流程、开源状态、算力估算和数据要求见 [复现指南](docs/REPRODUCTION_GUIDE_ZH.md)。字段约定见 [数据规范](docs/DATA_SCHEMA_ZH.md)。
从监督标签、质量控制、ATLAS12/SYNTHE 梯度库到 A100 正式训练的逐步执行清单见 [STEPS.md](STEPS.md)。
其中第 1.4 节规定了从当前 DR9 LRS 子集扩展到 LRS/MRS、多个 release 和全量推断的数据湖方案。
候选 `obsid` 的目录级命令和下载后预处理衔接也记录在 STEPS.md 第 1.4 节。

## 项目结构

```text
configs/                  可审计的预处理、训练和推断配置
data/external/            外部标签和 Kurucz 梯度谱（不入 Git）
data/interim/             清单、匹配表和中间结果（不入 Git）
data/metadata/            小型模板及字段说明
data/processed/           HDF5 训练集（不入 Git）
docs/                     中文复现说明与数据契约
models/                   模型检查点（不入 Git）
outputs/                  指标、日志和推断目录（不入 Git）
spectra/                  原始 LAMOST DR9 光谱（不入 Git）
src/ddpayne/              Python 包
tests/                    单元与小型集成测试
```

## 复现层级

1. `smoke-test`：验证代码路径，不产生科学结果。
2. 数据驱动基线：有高分辨率标签、无梯度库；可用于排错，不能称为完整 DD-Payne。
3. DD-Payne 复现：加入与 LAMOST 分辨率和归一化一致的物理梯度谱。
4. 论文级验证：重复观测、星团、外部巡天、梯度谱和 Cramer-Rao 界全套验证。
