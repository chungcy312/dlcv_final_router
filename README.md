# DLCV Final – Router Classifier

place router.pth @ checkpoint/router to inference

Run 
```bash
python router.py 
```
for demo

Run
```bash
python router.py path/to/json
```
for inference json file (with router.classifier)


batch = 256:
val_correct: 630
val_total: 630
Inference 630 questions in 0.0 min 2.594860553741455 sec
Average: 0.004118826275780087 sec each

Using router.Classify
accuracy: 1.0
Inference 630 questions in 0.0 min 8.902583360671997 sec
Average: 0.01413108469947936 sec each