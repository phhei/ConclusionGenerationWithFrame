# Strategies for framing argumentative conclusion generation

Code implementing techniques described by the paper, accepted at INLG2022.

_Generating an argumentative conclusion from  a set of textual premises is a challenging task,  due to a large range of possible conclusions.  In order to provide a conclusion generation  model with guidance towards generating conclusions from a certain perspective, we explore the impact of conditioning the model on information about the desired framing. We  experiment with conditioning generation via  generic frame classes as well as with so-called  issue-specific frames. Beyond conditioning the model on a desired frame, we investigate the impact of strategies to further improve the generated conclusion by i) an informative label smoothing method that dynamically smooths one-hot-encoded reference conclusion vectors as a regularization mechanism, and ii) a conclusion reranking strategy based on referenceless scores at inference time. We evaluate the benefits of our methods using metrics for automatic evaluation complemented with an extensive manual study. Our results show that frame-guided conclusion generation is beneficial: it increases the ratio of valid and novel conclusions by 23%-points compared to a baseline without frame information. Our work indicates that i) by injecting frame information, conclusion generation can be directed towards desired aspects and ii) at the same time it can be manually confirmed to yield more valid and novel conclusions._

## Requirements and Usage

We tested our code with Python V3.8. You have to run ``pip install -r requirements.txt`` beforehand.

### Main usage (generating conclusions given premises)

Run ``main.py``

Here you can specify different parameters at the start of the file:

- INPUT params (premises and reference conclusions)
- TRAINING parameters
- INFERENCE parameters
- OTHER PARAMS (logging stuff)

### For the reference-less scores: training models

#### Stance-Classifier based on NLI

Run ``NLIClassifier.py``

Control parameters at the start of the file, see "HYPERPARAMETERS"

#### (Generic)-Frame-Classifier

Run ``FrameClassifier.py``

## (Recommended) Datasets

- [Webis-argument-framing.csv](https://webis.de/data/webis-argument-framing-19.html) (recommended placing it in the project root folder)
- [Media-Frames-Dataset](https://github.com/dallascard/media_frames_corpus) (recommended placing it in _frame_sets/frame_datasets_)
  - Unfortunately, we're not allowed to share the plain data due to license issues. If you've the plain data, you need to convert it into a CSV with "argument_id|frame_id|frame|topic_id|topic|premise|conclusion"-columns. Please see [this repo](https://github.com/phhei/FramingNN) as reference
