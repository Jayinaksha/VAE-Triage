**Recall on artefact types NEVER seen in training (supervised models saw the other 6 types).**

| Model | bubble | pen ink | fold | JPEG | clean rejected |
|---|---|---|---|---|---|
| Logistic regression (supervised, 6 artefact types seen) | 0.13 | 0.64 | 0.41 | 0.46 | 0.041 |
| Random forest (supervised, 6 artefact types seen) | 0.07 | 0.64 | 0.24 | 0.22 | 0.062 |
| GMM K=1 (no artefact ever seen) | 0.61 | 0.83 | 0.91 | 0.96 | 0.068 |
| GMM K=6 (no artefact ever seen) | 0.64 | 0.85 | 0.94 | 0.99 | 0.072 |
