# v2 配对离线验证

这是自建离线环境的结果，不是官方模拟器演练或正式测试。
每一对使用相同源位置、频道、接收半径、朝向和固定位置误差规则；v1 保持原始默认参数。
两种算法经过同一运行包装器和独立 Benchmark 计时重放。失败案例保留，全清优先。

套件状态：`passed`；计划 30 对；实际记录 60 局。
源码 SHA-256：`07284746dbed216cd04bcf03f9645f874ce5dfcdbbdc584fe532d509d798f397`。

| 题型 | 算法 | 全清且验证通过 / 局数 | 每局平均时间的均值（秒/个） | 清除失败次数均值 | 最后清除后耗时均值（秒） |
| --- | --- | ---: | ---: | ---: | ---: |
| Q3 | v1 | 13/13 | 373.28 | 2.08 | 482.58 |
| Q3 | v2 | 13/13 | 327.04 | 2.92 | 172.88 |
| Q4 | v1 | 17/17 | 1003.48 | 67.47 | 3850.28 |
| Q4 | v2 | 17/17 | 795.95 | 2.00 | 428.88 |

均值是 mean(Tᵢ/Kᵢ)，含清除后确认成本；非完整日志的均值记为不可用。若有失败，时间不能单独用于宣称改进。

| 场景 | v1 全清验证 | v2 全清验证 | v1 秒/个 | v2 秒/个 | 配对时间变化 |
| --- | --- | --- | ---: | ---: | ---: |
| q3_legacy_20260910 | True | True | 441.15 | 377.51 | -14.4% |
| q3_legacy_20260911 | True | True | 312.57 | 255.31 | -18.3% |
| q3_legacy_20260912 | True | True | 422.80 | 415.25 | -1.8% |
| q3_stratified_n10 | True | True | 400.98 | 347.79 | -13.3% |
| q3_stratified_n11 | True | True | 409.30 | 351.84 | -14.0% |
| q3_stratified_n12 | True | True | 380.50 | 346.78 | -8.9% |
| q3_stratified_n13 | True | True | 405.89 | 327.52 | -19.3% |
| q3_stratified_n14 | True | True | 360.49 | 323.75 | -10.2% |
| q3_stratified_n15 | True | True | 366.12 | 282.74 | -22.8% |
| q3_stratified_n16 | True | True | 354.67 | 307.08 | -13.4% |
| q3_boundary | True | True | 352.56 | 459.04 | +30.2% |
| q3_cluster | True | True | 467.12 | 303.44 | -35.0% |
| q3_coincident | True | True | 178.50 | 153.50 | -14.0% |
| q4_legacy_20260910 | True | True | 1394.58 | 1021.73 | -26.7% |
| q4_legacy_20260911 | True | True | 508.06 | 671.05 | +32.1% |
| q4_legacy_20260912 | True | True | 1298.15 | 987.56 | -23.9% |
| q4_stratified_n10 | True | True | 1221.53 | 962.99 | -21.2% |
| q4_stratified_n11 | True | True | 1198.05 | 882.70 | -26.3% |
| q4_stratified_n12 | True | True | 996.52 | 811.59 | -18.6% |
| q4_stratified_n13 | True | True | 1038.96 | 812.92 | -21.8% |
| q4_stratified_n14 | True | True | 1127.12 | 753.07 | -33.2% |
| q4_stratified_n15 | True | True | 921.29 | 713.05 | -22.6% |
| q4_stratified_n16 | True | True | 603.49 | 626.52 | +3.8% |
| q4_boundary | True | True | 1147.81 | 1038.38 | -9.5% |
| q4_cluster | True | True | 1055.71 | 807.29 | -23.5% |
| q4_coincident | True | True | 715.16 | 622.50 | -13.0% |
| q4_outward_positive | True | True | 1275.19 | 728.03 | -42.9% |
| q4_outward_negative | True | True | 1231.99 | 721.87 | -41.4% |
| q4_tangent_positive | True | True | 617.49 | 683.99 | +10.8% |
| q4_tangent_negative | True | True | 708.11 | 685.91 | -3.1% |

负值表示 v2 更快。上述有限场景包含原 v1 种子以及分层和压力案例，不代表官方总体分布，
也没有跨随机案例构造官方配对效果。不同算法程序耗时在本机测得，不预测 Windows HTTP 性能。

每局证据位于同目录的场景子文件夹：metadata、events、decisions、offline_truth、metrics。
`verify_artifacts.py` 可独立核对计时、真值包含性和包围圆清除证据。
