The objective of this challenge is to arrange Christmas tree toys into the smallest packing arrangement possible&mdash;as defined by the size of a square bounding box around the trees&mdash;for toy counts of 1-200.

## Files

**sample_submission.csv** - a sample submission in the correct format
 - `id` - a combination of the `n`-tree count for the puzzle and the individual tree index within the puzzle
 - `x`, `y` - the 2-d coordinates of the tree; this point is defined at the center of the top of the trunk
 - `deg` - the rotation angle of the tree


This [Getting Started Notebook](https://www.kaggle.com/code/inversion/santa-2025-getting-started) implements a basic greedy algorithm (which was used to construct the `sample_submission.csv`), demonstrates collision detection, and provides a visualization function.

**Note:** The [metric](https://www.kaggle.com/code/metric/santa-2025-metric) for this competition has been designed to reasonably maximize floating point precision during calculations, but cannot guarantee precision beyond what is displayed on the leaderboard.