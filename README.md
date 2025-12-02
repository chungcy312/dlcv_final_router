## DLCV Final – Router Classifier

# Inference
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
for inference json file (using router.classifier)


# Performance
batch = 256:<br>
val_correct: 630<br>
val_total: 630<br>
Inference 630 questions in 0.0 min 2.594860553741455 sec<br>
Average: 0.004118826275780087 sec each<br>
<br>
Using router.Classify<br>
accuracy: 1.0<br>
Inference 630 questions in 0.0 min 8.902583360671997 sec<br>
Average: 0.01413108469947936 sec each<br>