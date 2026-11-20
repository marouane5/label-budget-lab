# Label Budget Lab

What should a classifier ask a human to label next? This experiment gives a handwritten-digit classifier a budget of 240 annotations and compares two ways to spend them. One draws images uniformly at random. The other requests the images with the highest predictive entropy.

At 240 labels, random selection averaged **90.47% accuracy** on images from unseen writers. Entropy selection averaged **88.02%**. Random selection won at that budget in each of the five paired runs.

## Replay the experiment

With Python 3.11–3.13, from the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python benchmark.py --fetch --out run
```

The first run downloads the original UCI ZIP into `data/` and verifies the two member files against SHA-256 checksums. For offline runs, pass `--archive path/to/optdigits.zip`. The benchmark runs on CPU with NumPy. Use `--seeds 3 5 7` or `--budgets 40 60 80 100` to explore another paired design. The first budget is the size of the shared starting batch.

## Experimental design

- **Writer-disjoint evaluation.** The acquisition pool contains 3,823 images from 30 writers. All 1,797 images from the other 13 writers remain in the original UCI test split. The 64 pixel features are divided by the known maximum of 16. Evaluation uses accuracy and macro-averaged F1 across the ten digits.
- **Paired budgets.** Each seed draws 40 initial images uniformly without labels. Both strategies receive exactly those same 40 images. Each round reveals 40 more labels, with no replacement. A simulated annotator reveals the UCI digit only after the strategy chooses its row.
- **Same classifier.** Each round fits a multinomial logistic regression from zero using 180 iterations of full-batch gradient descent, learning rate 0.4 and L2 penalty 0.002. The intercept is unpenalized. The entropy strategy selects the top 40 values of `-sum(p * log(p))` across the remaining pool, without updating within a batch. A stable sort resolves ties by training row number.
- **Separated evaluation.** The learning loop has access to training features and a label callback. It returns model snapshots for every budget. The evaluator scores these snapshots on the fixed test set after acquisition finishes. Five seeds change the initial labels and random draws. The variability describes acquisition randomness on this single test population.

## Learning curve

Mean test accuracy over seeds 17, 29, 43, 71 and 113. The last column is entropy minus random in percentage points.

| Total labels | Random | Entropy | Paired difference |
| ---: | ---: | ---: | ---: |
| 40 | 78.81% | 78.81% | 0.00 |
| 80 | 85.34% | 80.13% | −5.21 |
| 120 | 87.80% | 81.79% | −6.01 |
| 160 | 89.28% | 83.58% | −5.70 |
| 200 | 90.21% | 86.31% | −3.90 |
| 240 | 90.47% | 88.02% | −2.45 |

The largest mean gap occurred at 120 labels. The acquisition log offers one clue. In the first newly selected batch of 40, the most frequent digit accounted for an average of **51%** of entropy-selected images across the five seeds and **17%** of randomly selected images. For seed 113, entropy spent 31 of those 40 labels on the digit 8. These counts describe the observed class concentration. Isolating its effect on accuracy would require a separate experiment. Batch size, model confidence and the choice of dataset can all affect how well this strategy works.

### Inspect the evidence

- [Learning curves](results/summary.csv) include paired means, standard deviations and win counts for accuracy and macro-F1.
- [Per-run metrics](results/metrics.csv) contain every strategy, seed and budget.
- [Acquisition log](results/queries.csv) lists each newly labeled example by its zero-based row number in `optdigits.tra`. The `labels` column gives the total budget after that round.
- [Run settings](results/settings.json) record the NumPy version, budgets, seeds and data fingerprints.

## Sources

E. Alpaydin and C. Kaynak (1998) created *Optical Recognition of Handwritten Digits*. The original data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The acquisition log contains selected labels and source row IDs from that dataset. This project downloads the original archive from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). [Dataset DOI](https://doi.org/10.24432/C50P49).

Burr Settles (2009), *Active Learning Literature Survey*, section 3.1, describes entropy-based uncertainty sampling. [University of Wisconsin report](https://burrsettles.com/pub/settles.activelearning.pdf).
