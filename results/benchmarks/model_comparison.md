| Model | Params | Size | Precision | Recall | F1 | mAP50 | mAP50-95 | Latency | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo11n.pt | 2582737 | 5.23 MB | 0.7167 | 0.7875 | 0.7505 | 0.7353 | 0.3557 | 8.49 ms | 117.79 |
| yolo11s.pt | 9413961 | 18.30 MB | 0.7643 | 0.6548 | 0.7053 | 0.6163 | 0.3143 | 9.93 ms | 100.66 |
| yolo11m.pt | 20032345 | 38.65 MB | 0.7476 | 0.6907 | 0.7181 | 0.6488 | 0.3279 | 21.91 ms | 45.64 |

## Scientific Interpretation

`yolo11n.pt` has the highest mAP50-95. `yolo11n.pt` has the best aggregate recall. `yolo11n.pt` is fastest under the measured hardware condition. Inspect `per_class_metrics.csv`, especially `no_helmet` recall, before using the detector in safety-monitoring experiments. Class imbalance effects should be interpreted together with the dataset audit report.

## Trade-off Notes

- yolo11n.pt: mAP50-95=0.3557, latency=8.49 ms/image, FPS=117.79, size=5.23 MB
- yolo11s.pt: mAP50-95=0.3143, latency=9.93 ms/image, FPS=100.66, size=18.30 MB
- yolo11m.pt: mAP50-95=0.3279, latency=21.91 ms/image, FPS=45.64, size=38.65 MB

Recommendation: `yolo11n.pt` is selected by the documented rule: prioritize mAP50-95, inspect recall, and prefer the smaller/faster model when accuracy gains are marginal relative to latency and size.