# LAMOST DR9 DD-Payne 完整实施步骤

本文是本项目从“只有 LAMOST 光谱”推进到“有监督标签、物理梯度库、可审计训练结果”的执行清单。
它针对当前仓库的目录、字段和命令编写，默认后续在通过 SSH 连接的 Linux 服务器上使用 NVIDIA A100
训练。

本文不把下载到的巡天管线结果直接称为绝对真值。APOGEE/GALAH 标签本身有测量误差、选择效应、太阳丰度
标尺和管线系统误差；DD-Payne 的可信度来自统一标签定义、严格质量控制、物理梯度约束和独立验证的组合。

## 0. 最终目标和不可跳过的原则

最终训练输入必须同时包含：

1. LAMOST 归一化光谱和像素逆方差；
2. 每条光谱对应的 15 维监督标签：
   `teff`, `logg`, `vmic`, `fe_h`, `c_fe`, `n_fe`, `o_fe`, `mg_fe`, `al_fe`,
   `si_fe`, `ca_fe`, `ti_fe`, `cr_fe`, `mn_fe`, `ni_fe`；
3. 与训练波长网格、掩码、连续谱定义和 LAMOST LSF 完全一致的物理梯度库；
4. 恒星级分组的 train/validation/test 划分；
5. 原始目录版本、质量筛选、标签转换、梯度生成和训练配置的快照。

以下原则必须写入实验记录：

- 第一版建议只用 **APOGEE DR17 作为训练标签源**，GALAH DR3 先作为外部验证源。两个巡天不应把原始标签直接逐列拼接。
- 只有在 APOGEE 覆盖不足时，才考虑加入 GALAH；加入前必须用共同恒星估计每个标签的零点、尺度和参数依赖系统差异。
- 训练用的标签必须是光谱拟合定义一致的一套标签。校准后的物理参数可以保留用于验证，但不能无记录地和光谱拟合参数混用。
- 梯度谱必须经过与观测谱完全相同的 LSF 卷积、真空波长网格、静止系处理、连续谱归一化和像素质量规则。
- 训练范围之外的标签只能标记为外推或拒绝，不能当作正常测量发布。
- A100 主要用于 Payne 网络训练和标签反演；传统 ATLAS12/SYNTHE 通常是 CPU/文件 I/O 主导，不能假定换成 A100 后合成大气会线性加速。

## 1. 固化服务器环境

### 1.1 建立项目和日志目录

```bash
git clone <your-private-repository-url> DD-PayneRecurrence
cd DD-PayneRecurrence

mkdir -p data/external data/interim data/processed data/metadata
mkdir -p models outputs logs third_party
```

原始光谱、外部目录、合成光谱和中间 HDF5 不要提交到 Git。为每个实验建立独立目录，例如：

```text
outputs/2026-09-17-apogee15/
outputs/2026-09-17-apogee15/config.snapshot.yaml
outputs/2026-09-17-apogee15/metrics.jsonl
outputs/2026-09-17-apogee15/best.pt
```

### 1.2 检查 SSH、GPU、驱动和磁盘

```bash
ssh user@server
nvidia-smi
df -h
python3 --version
```

至少确认：

- A100 显存大小和 MIG 是否开启；如果使用整卡训练，确认没有被切成过小的 MIG 实例；
- NVIDIA 驱动与选定的 CUDA/PyTorch wheel 相容；
- 原始 LAMOST、APOGEE/GALAH、合成光谱和 HDF5 需要足够磁盘；
- 训练过程放在 `tmux`、`screen` 或调度系统作业中，不依赖 SSH 会话持续存在。

### 1.3 安装 Python 环境并记录版本

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 依据服务器驱动和 PyTorch 官方安装选择器选择 CUDA wheel。
# A100 不要照搬本地 RTX 5070 Ti 的 CUDA 版本命令。
python -m pip install -e ".[dev]"

python - <<'PY'
import sys, torch, astropy, numpy, scipy
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("astropy", astropy.__version__)
PY

git rev-parse HEAD > outputs/environment.git_commit.txt
pip freeze > outputs/environment.pip_freeze.txt
nvidia-smi -q > outputs/environment.nvidia-smi.txt
```

### 1.4 从当前 DR9 LRS 扩展到全 LAMOST 数据

当前仓库的 `spectra/dr9-v2.0-lrs-fits` 只是一个本地 DR9 LRS 子集，不是完整 LAMOST 数据集。后续的“全 LAMOST”应定义为按 `release x survey_mode x product_version` 管理所有公开数据产品，而不是把不同分辨率、不同管线版本的 FITS 文件直接拼成一个目录。

LAMOST 官方 DR9 v2.0 文档给出了约 1080 万条 LRS 光谱，并区分 LRS General Catalog、AFGK 参数表、多历元表和其他专题表；官方 DR9 页面同时提供 LRS/MRS 的目录入口。DR10 及以后版本也应按同样方式分别记录 release 和模式。参考入口：

- DR9 v2.0 总入口：<https://www.lamost.org/dr9/v2.0/>
- DR9 LRS/MRS 目录下载：<https://www.lamost.org/dr9/v2.0/catalogue>
- DR9 LRS 数据说明：<https://www.lamost.org/dr9/v2.0/doc/lr-data-production-description>
- DR10 MRS 数据说明：<https://www.lamost.org/dr10/v2.0/doc/mr-data-production-description>

#### 全量数据的正确获取顺序

不要先下载所有 LAMOST 光谱再寻找 APOGEE 交集。应按以下顺序建立可复现的候选索引：

1. 下载每个 release/mode 的官方总目录或可查询目录，并保存 release、版本、下载日期和 hash。
2. 统一成一个“观测记录表”，一行代表一次观测，至少包括：

   ```text
   lamost_release, survey_mode, product_version,
   obsid, uid, gp_id, designation,
   gaia_source_id, ra, dec, epoch, mjd,
   planid, spid, fiberid,
   snru, snrg, snrr, snri, snrz,
   spectrum_uri, local_path, checksum
   ```

3. 先把 APOGEE 训练标签的 `source_id` 与观测记录按 Gaia ID 或坐标匹配，得到目标 `obsid` 列表。
4. 只下载或挂载这些 `obsid` 对应的 LAMOST FITS，验证 checksum 后再进入预处理。
5. 完成第一版模型和外部验证后，再为全量推断建立分片下载队列。

当前 `ddpayne build-manifest` 读取已经落盘的 FITS 头，适合现有 DR9 LRS 子集；它还不是全 LAMOST 目录适配器。正式扩展前需要增加一个目录适配层，将官方 LRS/MRS 表中的 `spectrum_uri`、`obsid`、`gaia_source_id` 和 release 元数据写入观测记录表。这个适配层必须先于全量下载实现。

#### 建议的数据湖目录

```text
data/raw/lamost/
  dr9/lrs/catalogs/
  dr9/lrs/spectra/
  dr9/mrs/catalogs/
  dr10/lrs/catalogs/
  dr10/mrs/catalogs/
data/interim/lamost_inventory/
  dr9_lrs_observations.parquet
  dr9_mrs_observations.parquet
  all_releases_observations.parquet
data/processed/lamost/
  dr9_lrs_apogee_train.h5
  dr9_lrs_apogee_test.h5
  dr10_lrs_external_test.h5
  mrs_separate_model_input.h5
```

大型观测记录表应使用 Parquet/数据库分片，不要把所有 release 的索引长期维护成一个巨大 CSV。当前项目的 CSV/HDF5 路径用于开发样本；全量阶段需要增加 Parquet/SQLite/DuckDB 读取器和分片 HDF5/Zarr 写入器。

#### LRS、MRS 和不同 release 的边界

- **LRS**：当前 DD-Payne 网络的第一目标。DR9 LRS 约为 3700-9000 A、分辨率约 1800 的观测域；训练网格、LSF 和连续谱处理保持一套固定定义。
- **MRS**：不能直接和 LRS 拼接。MRS 波长段、分辨率、噪声和 LSF 均不同，建议先做独立 MRS 模型和独立梯度库；只有在验证域差异后才考虑条件化模型。
- **不同 release**：DR9、DR10 及以后可能有管线、波长、质量位和观测分布变化。默认将 release 作为 domain 元数据，并做“按 release 留出”的测试；不要把 release 字符串直接当普通元素标签输入网络。
- **不同观测产品**：重复观测保留为独立样本，但以稳定恒星 ID 分组切分；合并光谱只能作为额外数据产品，不能同时把原始重复谱和合并谱当作独立恒星。

推荐的扩展顺序是：`DR9 LRS -> DR10 LRS -> 其他 LRS release -> DR9/DR10 MRS`。每扩展一个域，都先做零点、S/N、LSF、波长覆盖和重复观测一致性报告。

#### 全 LAMOST 训练与全量推断的两种任务

必须把以下两个目标分开：

1. **跨巡天监督训练**：只需要下载与 APOGEE 高质量标签重叠的 LAMOST 观测，数量通常远小于全量。
2. **全巡天推断**：训练模型固定后，才对所有合格 LRS/MRS 观测分片推断；不要求所有全量光谱都拥有 APOGEE 标签。

这样既能用全 LAMOST 数据产出目录，也不会为了训练标签而下载和预处理数千万条无监督光谱。

#### 当前项目可直接执行的候选 obsid 命令

当前新增的 `build-candidates` 支持 LAMOST 官方目录的 CSV/FITS/Parquet 文件，以及 APOGEE/GALAH 标签表的 CSV/FITS/Parquet 文件。它先尝试 Gaia DR3/DR2 ID，再对未匹配行使用坐标最近邻；结果中的 `match_method`、`match_separation_arcsec` 和 `match_coordinate_conflict` 用于审计。

对于你当前已经下载的 `spectra/dr9-v2.0-lrs-fits`，可以先从 FITS 头生成目录，再执行同一个命令：

```bash
ddpayne build-manifest \
  --spectra-root spectra/dr9-v2.0-lrs-fits \
  --output data/metadata/spectra.csv

ddpayne build-candidates \
  --lamost-catalog data/metadata/spectra.csv \
  --labels data/external/allStar-dr17-synspec_rev1.fits \
  --output data/interim/lamost_apogee_candidates.csv \
  --quality apogee \
  --max-separation-arcsec 1.0
```

这条路径不需要重新下载当前本地 FITS；当前工作区已经有 `data/external/allStar-dr17-synspec_rev1.fits` 时可直接使用。由于当前 FITS 头通常只有 RA/DEC 而没有 Gaia ID，这时程序会自动使用坐标回退，并把匹配角距离写入输出。

注意：`data/external/lamost_full_lrs_catalog.fits` 和下面的
`data/external/lamost_dr9_lrs_general.fits` 都只是“官方 LAMOST 目录下载后保存到本地”的示例路径，项目不会自动生成它们。若文件尚未下载，直接把它们作为 `--lamost-catalog` 会报 `FileNotFoundError`。当前本地光谱请使用上面的 `data/metadata/spectra.csv`；只有在准备全量巡天目录时，才下载官方目录并把命令中的路径替换成真实文件。

```bash
ddpayne build-candidates \
  --lamost-catalog <官方LAMOST目录文件> \
  --labels data/external/allStar-dr17-synspec_rev1.fits \
  --output data/interim/lamost_dr9_lrs_apogee_candidates.csv \
  --quality apogee \
  --gaia-version auto \
  --max-separation-arcsec 1.0
```

输出表至少包含 `obsid`、`source_id`、`lamost_source_id`、标签列、`match_method` 和 `match_separation_arcsec`。如果官方目录提供 `fitsname`/`spectrum_uri`，程序还会保留为 `path`/`spectrum_uri`，可以据此生成下载清单。当前 `build-candidates` 不会自动从 LAMOST 服务器下载光谱，也不会伪造本地 `path`。

命令还会在同目录写出同名的 `.obsid.txt`，每行一个候选 `obsid`，可以直接粘贴到 LAMOST 官方批量下载工具或作为你自己的下载脚本输入。

下载候选光谱后，把 `path` 更新为项目内真实相对路径，并先抽查：

```bash
ddpayne inspect-data \
  --spectra-root data/raw/lamost/dr9/lrs/spectra \
  --limit 100

ddpayne prepare --config configs/preprocess_lamost.yaml
```

`configs/preprocess_lamost.yaml` 的 `matched_labels` 可以直接指向候选表，但候选表中的 `path` 必须已经存在，且 `source_id` 必须是标签源的恒星级 ID。若 `match_coordinate_conflict` 为真，先人工核查或删除对应行；ID 匹配优先级高于角距离，但大角距离通常提示 epoch、ID 版本或邻近源问题。

## 2. 获得监督标签

### 2.1 推荐优先级

本项目建议采用以下顺序：

1. **APOGEE DR17 `allStar`**：作为第一版主训练标签源。它包含天体物理参数、元素丰度、误差和 ASPCAP/STAR 质量信息，且和论文方法中的高分辨率监督标签用途最接近。
2. **GALAH DR3 `GALAH_DR3_main_allstar_v2.fits`**：作为独立验证源，或在 APOGEE 与 LAMOST 重叠严重不足时作为第二标签源。
3. 其他高分辨率巡天：只能在明确太阳丰度标尺、标签定义、质量位和坐标元数据后接入。

不要把 LAMOST 自身的低分辨率管线标签当作本项目的高分辨率监督标签。它们可以作为质量筛选或外部比较，但会把目标模型训练成 LAMOST 管线的复制器。

### 2.2 下载 APOGEE DR17

官方数据访问页面：

`https://www.sdss4.org/dr17/irspec/spectro_data/`

优先下载 `allStar-dr17-synspec_rev1.fits`。完整 `allStar` 约为数 GB；先做一轮小规模验证时，可以下载或查询 `allStarLite`，但正式 15 标签训练应确认 lite 文件包含所需的所有命名丰度、误差和质量位。

在服务器上按官方 SAS 目录下载，并保留文件大小、发布日期和校验和：

```bash
cd data/external
wget -c \
  "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits" \
  -O allStar-dr17-synspec_rev1.fits
sha256sum allStar-dr17-synspec_rev1.fits \
  > allStar-dr17-synspec_rev1.fits.sha256
```

如果服务器不能下载整个目录，可以使用 SDSS CAS/SAW 先按天球区域查询候选星，再下载一个只包含候选星的本地 CSV/FITS。无论采用哪种方式，都必须保存原始 SQL、查询日期、表名和 release 版本。

### 2.3 下载 GALAH DR3

官方目录页：

`https://www.galah-survey.org/dr3/the_catalogues/`

推荐下载：

```bash
cd data/external
wget -c \
  "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_main_allstar_v2.fits" \
  -O GALAH_DR3_main_allstar_v2.fits
sha256sum GALAH_DR3_main_allstar_v2.fits \
  > GALAH_DR3_main_allstar_v2.fits.sha256
```

`allstar` 是每颗恒星一条记录的清洁目录，适合训练样本；`allspec` 是每次光谱一条记录，适合研究重复观测，但不能直接和 `allstar` 混用而不做恒星级分组。

### 2.4 先检查目录列名，不要凭记忆映射

```bash
python - <<'PY'
from astropy.io import fits

for path in [
    "data/external/allStar-dr17-synspec_rev1.fits",
    "data/external/GALAH_DR3_main_allstar_v2.fits",
]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        print("\\n", path, len(data))
        print([name for name in data.names if any(key in name.upper() for key in
              ["RA", "DEC", "TEFF", "LOGG", "VMIC", "FE_H", "C_FE", "FLAG"])][:100])
PY
```

## 3. 统一标签表到项目契约

### 3.1 APOGEE 字段映射

输出文件建议为 `data/external/apogee_dr17_labels.csv`，至少包含：

| 项目字段 | APOGEE DR17 常用字段 | 说明 |
|---|---|---|
| `source_id` | `APOGEE_ID` | 恒星级主键；不要用文件行号 |
| `ra`, `dec` | `RA`, `DEC` | degree |
| `teff` | `TEFF_SPEC` 或已明确版本的 `TEFF` | 记录选用原因和版本 |
| `logg` | `LOGG_SPEC` 或已明确版本的 `LOGG` | 不要和校准值无记录混用 |
| `vmic` | `VMICRO` | km/s |
| `fe_h` | `FE_H` | dex |
| `c_fe` ... `ni_fe` | `C_FE`, ..., `NI_FE` | dex，相对 Fe |

APOGEE 官方文档建议优先使用命名标签和相应误差，并检查 `ASPCAPFLAG`、`STARFLAG` 以及每个元素的 flag。`[C/M]`、`[N/M]` 是拟合参数，不要直接当成项目要求的 `[C/Fe]`、`[N/Fe]`。

同时保留以下审计字段，不要在导出 CSV 时丢弃：

```text
apogee_id, release, field, telescope, aspcap_version,
aspcapflag, starflag, extratarg, snr,
teff_err, logg_err, vmicro_err, fe_h_err,
c_fe_err, ..., ni_fe_err,
c_fe_flag, ..., ni_fe_flag, label_source
```

缺测值、上限值、网格边缘值和 `-999` 这类 sentinel 必须统一转换为 NaN，不能让它们进入训练。

### 3.2 GALAH 字段映射

输出文件建议为 `data/external/galah_dr3_labels.csv`。以 `GALAH_DR3_main_allstar_v2` 的实际 schema 为准，常用字段通常包括：

```text
source_id = dr3_source_id 或稳定的 star_id
ra, dec
teff, logg, vmic, fe_h
c_fe, n_fe, o_fe, mg_fe, al_fe, si_fe,
ca_fe, ti_fe, cr_fe, mn_fe, ni_fe
```

GALAH 的元素可用性和字段命名必须以 `https://www.galah-survey.org/dr3/table_schema/` 为准。不存在或质量不足的元素不能用 0 填充；应先生成元素覆盖率表，再决定训练 15 标签还是减少标签集合。

保留：

```text
sobject_id, dr3_source_id, star_id, release,
flag_sp, flag_fe_h, flag_<element>,
e_teff, e_logg, e_vmic, e_fe_h, e_<element>,
label_source
```

### 3.3 标签定义决策记录

在 `data/metadata/label_definition.yaml` 中固定：

```yaml
label_names: [teff, logg, vmic, fe_h, c_fe, n_fe, o_fe, mg_fe, al_fe, si_fe, ca_fe, ti_fe, cr_fe, mn_fe, ni_fe]
temperature_definition: APOGEE_TEFF_SPEC
gravity_definition: APOGEE_LOGG_SPEC
abundance_definition: named_X_FE
solar_scale: record_the_catalogue_scale
velocity_unit: km/s
abundance_unit: dex
```

如果实际列名与示例不同，修改这份定义文件和转换脚本，不要只在一次性 Notebook 中隐式修改。

## 4. 标签质量控制

### 4.1 APOGEE 初始质量切

第一版建议采用严格切选，后续用验证结果评估是否放宽：

1. `ASPCAPFLAG` 不设置 `STAR_BAD`、`CHI2_BAD`、`GRIDEDGE_BAD` 相关位；不要简单把所有非零 bit 都删除，因为其中有些只是 informational/warning。
2. `STARFLAG` 去除明显坏像素、严重散射、仪器问题和不可信 RV 的对象。
3. `EXTRATARG == 0` 作为主样本起点；特殊目标另存，不直接混入第一版。
4. S/N 先采用 `SNR >= 70` 附近的高质量样本作为训练起点；之后可按标签和元素分别评估更低 S/N。
5. 每个使用的元素都要求对应 abundance flag 通过，误差有限且小于预设阈值。
6. 对 `teff`, `logg`, `vmic`, `[Fe/H]` 和元素丰度设置物理范围，并记录剔除数量。
7. 删除明显双星、快速自转或宽线目标，除非已经决定把旋转作为额外标签。

这些条件要输出为 `data/interim/apogee_quality_rejections.csv`，每行保存 `source_id`、失败规则和原始 flag。

### 4.2 GALAH 初始质量切

1. `flag_sp`、`flag_fe_h` 和使用元素的元素 flag 先要求通过；flag 的具体数值含义以 DR3 best practices 和 schema 为准。
2. 要求 `teff/logg/fe_h` 及选用元素的值、误差有限且不为 sentinel。
3. 删除明显的低 S/N、宽线、高旋转、双星和网格边缘对象。
4. `allstar` 作为每颗星一行的主样本；`allspec` 只用于重复观测一致性验证，不能让同一颗星跨 train/test。

### 4.3 覆盖率检查

生成并保存：

- 每个标签的有限值比例、误差分布和质量 flag 直方图；
- `Teff-logg`、`Teff-[Fe/H]`、`logg-[Fe/H]` 分布；
- 15 维完整标签样本数；
- 每个元素在 LAMOST 交叉匹配样本中的覆盖数；
- 训练集、验证集和测试集的相同分布比较。

如果 15 维完整样本过少，先训练一个 4 标签或 8 标签验证模型，再逐步增加元素。不要为了保留样本而用中位数、零值或模型预测值填补监督标签。

## 5. 构建 LAMOST 与标签目录的交叉匹配

### 5.1 生成 LAMOST 清单

在项目根目录执行：

```bash
source .venv/bin/activate
ddpayne inspect-data --spectra-root spectra/dr9-v2.0-lrs-fits --limit 1000
ddpayne build-manifest \
  --spectra-root spectra/dr9-v2.0-lrs-fits \
  --output data/metadata/spectra.csv
```

检查 `ra`, `dec`, `z`, `z_err`, `snrg`, `class` 的缺失率。当前程序在没有更稳定 ID 时会临时使用 FITS 头中的 `DESIG`，交叉匹配后必须用高分辨率巡天的稳定 `source_id` 作为恒星级分组键。

### 5.2 首轮本地天球匹配

先分别匹配 APOGEE 和 GALAH：

```bash
ddpayne match-labels \
  --manifest data/metadata/spectra.csv \
  --labels data/external/apogee_dr17_labels.csv \
  --output data/interim/lamost_apogee_matched.csv \
  --max-separation-arcsec 1.0

ddpayne match-labels \
  --manifest data/metadata/spectra.csv \
  --labels data/external/galah_dr3_labels.csv \
  --output data/interim/lamost_galah_matched.csv \
  --max-separation-arcsec 1.0
```

当前实现使用 Astropy 最近邻天球匹配。正式目录还要做以下审计：

1. 保存最近邻角距离和第二近邻角距离；
2. 对高密度区域拒绝近邻不唯一、第二近邻过近或多个 LAMOST 光谱指向一个标签源但无法确认的对象；
3. 如果有 Gaia DR3 `source_id`，优先利用 Gaia 与 APOGEE/GALAH 的官方交叉信息，再回连到 LAMOST，而不是只依赖名字字符串；
4. 对高自行星考虑把坐标传播到观测 epoch；
5. 抽查 100-500 条匹配的光谱、RA/DEC、S/N、Teff、logg 和分离角；
6. 输出匹配率、按观测夜/天空位置的匹配率和冲突数。

不要因为匹配率低就直接把半径扩大到 3-5 arcsec。先检查坐标 epoch、LAMOST `DESIG`、重复观测和高自行造成的系统误差。

## 6. 选择训练标签源和统一尺度

### 6.1 推荐的第一条路线：APOGEE 单源训练

先用 `lamost_apogee_matched.csv` 训练一个完整 15 标签版本。这样可以避免两个巡天的零点、太阳丰度和标签定义差异污染第一版网络。GALAH DR3 用于：

- 未参与训练的外部标签验证；
- 检查 `Teff/logg/[Fe/H]` 的系统偏差；
- 检查元素丰度随参数的趋势；
- 评估模型是否只是复制 APOGEE 的系统误差。

### 6.2 合并 APOGEE 和 GALAH 的条件

只有以下条件全部满足时才合并：

1. 同一标签的物理定义和太阳丰度标尺已明确；
2. 两个目录对同一批共同恒星有足够重叠；
3. 已按 `Teff/logg/[Fe/H]` 或更完整参数拟合每个标签的零点和趋势差异；
4. 差异校正的系数、训练数据和残差已保存；
5. 合并后的 `label_source`、误差和质量位仍然可追溯；
6. 留出一份完全不参与校正的共同恒星进行检验。

一个可审计的尺度校正形式是：

```text
label_galah_corrected = label_galah
                         + a0
                         + a1 * (teff - teff0)
                         + a2 * (logg - logg0)
                         + a3 * (fe_h - feh0)
```

实际模型可以更复杂，但必须防止高阶多项式把标签噪声拟合进去。没有足够共同恒星时，不合并。

## 7. 设计训练、验证和测试样本

### 7.1 恒星级分组

使用高分辨率目录的恒星 `source_id` 作为 group key：同一恒星的所有 LAMOST 重复观测只能进入一个 split。当前 `prepare` 会按照 `source_id` 做稳定划分，不能改成按文件随机切分。

推荐起点：

```text
train       80%
validation  10%
test        10%
```

### 7.2 训练覆盖

训练星不要只按总数随机抽取。要检查：

- `[Fe/H]` 的金属丰度尾部；
- 低温、高温、矮星、巨星和亚巨星；
- 每个元素的有效标签数量；
- LAMOST S/N、观测夜和光谱覆盖；
- 标签空间的凸包边界。

建议先从 3,000-20,000 颗高质量重叠星开始。若当前 14 个观测夜的 LAMOST 子集重叠不足，应扩大到完整 DR9 或追加其他夜，而不是用少量样本强行训练 15 维网络。

### 7.3 防止数据泄漏

以下内容均不能使用测试集来调参：

- 标签零点校正；
- 梯度权重；
- 连续谱宽度；
- LSF 参数；
- S/N 质量阈值；
- 模型隐藏层、学习率和训练步数。

星团可以作为最终外部验证，也可以另留一个星团作为开发验证，但不能一边看结果一边用它调整训练配置。

## 8. 制作 Kurucz 物理梯度库

### 8.1 工具和许可证记录

保留以下信息：

- ATLAS12、SYNTHE 原始代码下载地址、版本、编译器和编译选项；
- 线表版本、分子线表版本、solar abundance 文件；
- 运行脚本和输入模板；
- 每个合成谱的参数 JSON；
- 生成机器、CPU 核数、内存和运行时间。

Kurucz 官方入口：`https://kurucz.harvard.edu/`。

可以参考 `https://github.com/tingyuansen/kurucz` 的 Python 封装和 ATLAS12/SYNTHE 工作流，但最终科学结果必须验证封装是否调用了期望的原始代码、线表和边界条件。不要把封装器的默认值当成论文的默认值。

### 8.2 确定标签到大气输入的转换

对每一个参考点建立一份完整的物理参数文件：

```text
Teff [K]
logg [dex]
vmic [km/s]
[Fe/H] [dex]
[C/Fe], [N/Fe], [O/Fe], ... [Ni/Fe] [dex]
solar abundance scale
model atmosphere geometry / convection settings
```

转换关系必须明确：

```text
[X/H] = [Fe/H] + [X/Fe]
log_epsilon(X) = log_epsilon_sun(X) + [X/H]
```

`solar abundance scale` 一旦固定，不能在生成一半梯度库时更换。将该关系和太阳丰度表写进 `data/metadata/label_definition.yaml`。

### 8.3 选择参考点

先用训练标签的稳健范围，例如 1% 和 99% 分位数加小边界，而不是盲目使用超出训练集的理论范围：

```text
Teff: 由实际训练样本决定
logg: 由实际训练样本决定
[Fe/H]: 由实际训练样本决定
vmic: 由实际训练样本决定
[X/Fe]: 由每个元素的实际覆盖决定
```

建议分两级：

1. **开发级**：16-32 个覆盖凸包的参考点，先验证流程和梯度方向；
2. **正式级**：约 101 个参考点或经覆盖率证明的等价设计，用于降低参数空间内的梯度偏差。

若有 `R` 个参考点、`L=15` 个标签，中心差分至少需要：

```text
R * (1 + 2 * L)
```

组模型光谱。16 个点约 496 组，101 个点约 3131 组。若每个标签扰动都重新计算 ATLAS12 大气，这个步骤应放在 CPU 集群或调度系统中分批运行。

### 8.4 为每个参考点生成基准和扰动谱

每个参考点执行：

1. 用 ATLAS12 计算基准大气结构；
2. 用 SYNTHE 在高分辨率、真空波长上计算基准光谱；
3. 对每个标签分别生成 `label + step/2` 和 `label - step/2`；
4. 对 `Teff`、`logg`、`vmic`、`[Fe/H]` 等改变大气结构的标签，重新运行需要的 ATLAS12/SYNTHE 流程；
5. 对元素丰度扰动，使用能正确处理逐元素丰度和线不透明度的 ATLAS12/SYNTHE 设置；不能只在文件名中改变丰度而复用不相容的大气结构；
6. 对每个输出保存参数、输入文件哈希、运行状态和异常信息。

开发阶段推荐的差分步长示例：

```text
Teff:  50-100 K
logg:  0.05-0.10 dex
vmic:  0.05-0.10 km/s
abundances: 0.02-0.05 dex
```

这些不是固定真值。必须用步长减半测试：如果梯度变化很大，说明步长太大、数值噪声太强、模型网格边缘不稳定，或 ATLAS12/SYNTHE 的运行没有保持一致。

### 8.5 高分辨率谱到 LAMOST 观测空间

每一个基准谱和扰动谱都必须执行相同的观测算子：

1. 使用真空波长；
2. 使用 LAMOST 的波长依赖 LSF 卷积；
3. 先从平均 LSF 开发，正式结果再考虑板、光纤、观测夜或目标相关 LSF；
4. 重采样到项目的公共对数波长网格；
5. 应用和观测谱完全一致的波段裁切、Balmer/DIB 排除和无效像素规则；
6. 应用同样的 50 Å Gaussian 伪连续谱归一化；
7. 保存卷积前后波长、LSF 版本和归一化配置。

当前 LAMOST COADD 文件主要提供 `FLUX/IVAR/WAVELENGTH/ANDMASK/ORMASK`，不代表其中已经有可直接用于最终梯度库的逐星 LSF。若只能获得平均 LSF，必须在论文和结果元数据中明确这是近似，并把 LSF 误差列为系统误差来源。

### 8.6 计算中心差分梯度

对参考点 `r`、标签 `l` 和像素 `p`，保存：

```text
gradient[r, l, p] =
    (normalized_flux_plus[r, l, p]
     - normalized_flux_minus[r, l, p]) / step[l]
```

梯度单位是“归一化通量 / 物理标签单位”。`Teff` 的梯度按 K，丰度梯度按 dex，`vmic` 梯度按 km/s。不要先按训练集标准差缩放后再写入 NPZ；当前训练代码会在模型内部处理标签缩放，但梯度库仍必须按物理单位保存。

### 8.7 写出当前项目要求的 NPZ

最终文件：

```text
data/external/kurucz_lamost_gradients.npz
```

必须包含：

```text
labels       float32 [R, 15]
gradients    float32 [R, 15, P]
steps        float32 [15]
wavelength   float64 [P]
label_names  string  [15]
```

其中 `P` 必须和 `configs/preprocess_lamost.yaml` 的公共网格完全一致，`label_names` 必须和 HDF5 的顺序逐字一致。当前数据契约和校验逻辑见 [docs/DATA_SCHEMA_ZH.md](docs/DATA_SCHEMA_ZH.md)。

## 9. 梯度库科学质量检查

在允许正式训练前，必须逐项通过：

1. 所有数组 shape、dtype、波长顺序和 `label_names` 一致；
2. 没有 NaN、Inf、全零梯度或异常尖峰；
3. 步长减半后，主要吸收线区域的梯度方向稳定；
4. Fe、Mg、Ca、Ti 等应在预期吸收线附近呈现明显响应；
5. 不同标签的梯度不能全部呈现相同形状；
6. 梯度在参考点附近的有限差分重建误差可接受；
7. 物理标签扰动在 ATLAS12/SYNTHE 输出中确实生效；
8. 卷积前后谱线宽度符合 LAMOST 分辨率；
9. 连续谱处理不会把大段谱线归一化为 NaN；
10. 抽取至少 20 个参考点和 20 个波段做人工图形检查。

如果梯度库的物理量纲、label order 或波长网格不匹配，当前训练程序会拒绝启动。不要关闭这一校验。

## 10. 生成监督 HDF5

确认 `data/interim/lamost_apogee_matched.csv` 已经完成质量筛选和字段标准化后，复制或软链接为预处理配置指定的文件，或直接修改配置：

```bash
cp data/interim/lamost_apogee_matched.csv \
   data/interim/lamost_highres_matched.csv

ddpayne prepare --config configs/preprocess_lamost.yaml
```

先不要直接跑 125,075 条全量数据。建议先生成一个 100-1000 条的开发子集，检查：

- 通过和拒绝数量；
- `valid_fraction` 分布；
- 归一化光谱是否存在负值、异常尖峰或平坦为 1 的长区间；
- 静止系吸收线位置；
- 训练/验证/测试的恒星级比例；
- 每个标签的范围和缺失情况。

确认开发子集后再正式生成 `data/processed/lamost_training.h5`。HDF5 生成后保存：

```bash
sha256sum data/processed/lamost_training.h5 \
  > data/processed/lamost_training.h5.sha256
```

`prepare` 同时写出 `data/interim/preprocess_rejections.csv` 和
`data/interim/preprocess_summary.json`。拒绝表保留 `obsid`、高分辨率巡天
`source_id`、LAMOST `SNRG`、目标类型、拒绝阶段和原因；JSON 汇总记录输入数、
接收数、拆分数以及各拒绝原因的计数。低 S/N 或非恒星记录是质量控制结果，
不能为了扩大样本量直接改成接收。

当前本地子集的基线结果是 456 个 APOGEE-LAMOST 候选中接收 79 个：65 个训练、
6 个验证、8 个测试；376 个因 `SNRG < 20` 拒绝，1 个因 `class != STAR`
拒绝。79 个源的 `source_id` 均唯一且没有跨拆分泄漏。这批数据只能验证流水线，
不能作为可信的 15 标签丰度模型训练集。

在没有 Kurucz 梯度库时，可用真实预处理数据运行三步 CPU smoke test：

```bash
ddpayne train --config configs/train_local_smoke.yaml
```

该配置明确关闭物理梯度约束，只验证 HDF5、DataLoader、模型、优化器、验证和
checkpoint 写出是否连通；输出不能作为科学模型使用。

## 11. 在 A100 上启动第一次训练

### 11.1 A100 配置起点

复制训练配置为服务器实验配置，例如 `configs/train_apogee_a100.yaml`，建议从以下值开始：

```yaml
dataset: data/processed/lamost_training.h5
run_dir: outputs/2026-09-17-apogee15
device: cuda

model:
  hidden_size: 40
  pixel_chunk_size: 1024

optimization:
  batch_size: 512
  max_steps: 10000
  learning_rate_start: 0.01
  learning_rate_end: 0.0001
  mixed_precision: true
  num_workers: 4

gradient_regularization:
  path: data/external/kurucz_lamost_gradients.npz
  required: true
  apply_every_steps: 1
  reference_batch_size: 8
```

A100 显存充足时可尝试论文中的 batch 512；`pixel_chunk_size` 和 `reference_batch_size` 仍需以 `nvidia-smi` 的峰值显存为准。当前实现对观测重建使用 FP16 autocast，物理梯度有限差分使用 FP32，不能为了省显存把梯度正则强制放到 FP16。

### 11.2 启动顺序

```bash
source .venv/bin/activate

ddpayne smoke-test --work-dir outputs/smoke_a100

ddpayne train \
  --config configs/train_apogee_a100.yaml \
  2>&1 | tee logs/train_apogee15.log
```

正式训练前必须确认 smoke test 使用的是当前服务器的 CUDA 环境，而不是 CPU fallback。训练开始后保存：

- 配置快照；
- Git commit；
- `pip freeze`、`nvidia-smi -q`；
- 数据和梯度库 hash；
- `metrics.jsonl`；
- `best.pt` 和 `last.pt`。

### 11.3 调度系统示例

如果服务器使用 Slurm，可将下面内容改成实际 partition、account 和 module：

```bash
#!/bin/bash
#SBATCH --job-name=ddpayne-apogee
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
source .venv/bin/activate
python -m ddpayne train --config configs/train_apogee_a100.yaml \
  2>&1 | tee "logs/${SLURM_JOB_ID}.log"
```

## 12. 训练过程中的验收闸门

### 闸门 A：单批过拟合

用 32-128 颗星、少量参考梯度点和 100-500 步训练，要求：

- reconstruction loss 明显下降；
- gradient loss 不出现 NaN；
- 读取 `best.pt` 后结果一致；
- 网络能重建训练小批次而不是输出恒定连续谱。

### 闸门 B：无梯度和有梯度对比

执行两组完全相同的训练：

1. `required: false`，仅作 data-driven baseline；
2. `required: true`，完整 DD-Payne。

比较验证残差、标签偏差和物理梯度误差。没有梯度库的结果必须明确标记为 baseline，不能写成元素丰度最终目录。

### 闸门 C：独立测试

只在配置、梯度权重和模型选择都固定后，运行 test split。至少报告：

- 每个标签的 bias、MAD、RMSE；
- 按 S/N、Teff、logg、[Fe/H] 分箱的误差；
- 每个元素的训练集覆盖和边界比例；
- 重复 LAMOST 观测的一致性；
- APOGEE 训练标签与 GALAH 未参与训练标签的比较；
- 星团内部丰度散布；
- 模型梯度与物理梯度的逐标签、逐波段差异。

## 13. 全量标签反演

先从 100-1000 条光谱开始：

```bash
ddpayne infer --config configs/infer_lamost.yaml
```

确认以下字段后再扩大到全量：

```text
label columns
reduced_chi2
valid_pixels
boundary_flag
qflag_chi2
status
error
```

正式运行时：

1. 固定并记录 checkpoint hash；
2. 按日期或文件分片推断，避免单个作业失败导致全部重跑；
3. 每片输出独立 metadata JSON；
4. 汇总 rejected 数量、边界比例、chi2 分布和训练凸包外比例；
5. 只对质量位通过且位于训练覆盖内的目标发布元素丰度。

## 14. 发布前归档和复现包

最终至少归档：

```text
原始 LAMOST 文件清单和 hash
APOGEE/GALAH 原始文件名、release、下载日期和 hash
标签转换脚本和 label_definition.yaml
质量筛选规则、拒绝表和覆盖率报告
交叉匹配表、角距离和冲突处理报告
LSF 文件/版本和卷积脚本
ATLAS12/SYNTHE 版本、线表、输入模板和运行日志
梯度库 NPZ、生成配置和 QA 报告
HDF5 数据集和 hash
训练配置、Git commit、pip freeze、nvidia-smi
best.pt、last.pt、metrics.jsonl
测试报告和最终推断质量统计
```

建议为每次正式实验建立一个 manifest：

```yaml
experiment: 2026-09-17-apogee15
label_source: APOGEE_DR17
gradient_library: kurucz_lamost_gradients_v001.npz
lamost_release: DR9_v2.0_LRS
lsf_definition: record_file_or_average_model
solar_scale: record_exact_table
git_commit: record_hash
dataset_sha256: record_hash
gradient_sha256: record_hash
```

## 15. 推荐的实际执行顺序

为了降低风险，建议按以下五个里程碑执行：

### M1：APOGEE 单源小样本

- 下载并标准化 APOGEE DR17；
- 生成 LAMOST 清单并完成 1 arcsec 匹配；
- 确认至少有数百颗高质量、15 标签完整的重叠星；
- 完成预处理和可视化，不生成科学结果。

### M2：16-32 点梯度开发库

- 固定太阳丰度、label order 和差分步长；
- 生成少量参考点和全部 15 个标签扰动；
- 完成中心差分、卷积、归一化和 NPZ QA；
- 运行 32-128 星过拟合。

### M3：APOGEE 训练集扩展

- 扩大到 3,000-20,000 颗星；
- 重新设计参考点覆盖；
- 在 A100 上运行无梯度/有梯度对照；
- 固定超参数后只使用 test split 做最终评估。

### M4：GALAH 外部验证

- 下载并标准化 GALAH DR3 allstar；
- 单独交叉匹配和质量筛选；
- 不参与训练地比较标签偏差、元素趋势和参数边界；
- 如确实需要合并，先完成尺度校正和独立验证。

### M5：正式梯度库和全量推断

- 将开发级梯度库升级到正式参考点数量；
- 明确平均 LSF 的系统误差，或引入更细的 LSF 模型；
- 固定 checkpoint 和所有输入 hash；
- 分片推断完整 DR9，并发布带质量位和外推标志的目录。

## 16. 不能宣布“可信”的情况

出现以下任一情况时，只能称为软件或数据驱动基线，不能称为可信元素丰度模型：

- 没有高分辨率监督标签；
- 只有 APOGEE/GALAH 标签，没有物理梯度库；
- 梯度库没有经过 LAMOST LSF 和连续谱处理；
- 训练和测试包含同一恒星的重复观测；
- 标签超出训练凸包却没有外推标志；
- 未检查 APOGEE 与 GALAH 的零点差异；
- 未报告元素缺测、质量 flag 和标签误差；
- 仅凭训练 loss 下降，没有独立光谱和星团验证；
- 用 0、中位数或其他模型预测值填充缺失的监督标签。

## 参考资料

- APOGEE DR17 数据访问：<https://www.sdss4.org/dr17/irspec/spectro_data/>
- APOGEE DR17 ASPCAP 丰度说明：<https://www.sdss4.org/dr17/irspec/abundances/>
- APOGEE DR17 参数说明：<https://www.sdss4.org/dr17/irspec/parameters>
- APOGEE 质量位：<https://www.sdss4.org/dr17/algorithms/bitmasks/>
- GALAH DR3 目录：<https://www.galah-survey.org/dr3/the_catalogues/>
- GALAH DR3 schema：<https://www.galah-survey.org/dr3/table_schema/>
- Kurucz 官方主页：<https://kurucz.harvard.edu/>
- 可参考的 ATLAS12/SYNTHE Python 工作流：<https://github.com/tingyuansen/kurucz>
- Xiang et al. 2019, LAMOST DD-Payne：<https://arxiv.org/abs/1908.09727>
- Zhang et al. 2024, DESI DD-Payne：<https://arxiv.org/abs/2402.06242>
