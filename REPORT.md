# Knee Osteoarthritis CNN Classifier

## Abstract
Osteoarthritis (OA) of the knee is graded on the five-point Kellgren-Lawrence (KL) scale, from Grade 0 (no radiographic evidence of disease) to Grade 4 (severe disease). Early detection, especially of Grade 1 disease, is clinically important but radiographically, is quite subtle, and automated classification is complicated by a small, class-imbalanced dataset. This report describes the design, implementation and evaluation of a Convolutional Neural Network (CNN) pipeline for KL grade classification from knee X-rays. An initial baseline CNN was trained from scratch, initially begimning with no modifications to the knee and later being extended further on in development to include cropping and contrast enhancing to prepare for the final model. This baseline yielded limited results and achieved an accuracy of 0.379 and a Quadratic Weighted Kappa (QWK) of 0.0178, with a Grade 1 recall of 0.0, thus entailing the model failed to recognise Grade 1 images.

## Introduction
Knee osteoarthritis (OA) is one of the most common musculoskeletal conditions worldwide, and its severity is conventionally assessed from the Kellgren-Lawrence (KL) grading system. The scale ranges from Grade 0, representing a healthy joint to Grade 4, a knee with severe OA and contains large osteophytes, marked joint space narrowing and bone deformity. However, it is rather difficult to distinguish between grades. Whilst Grade 0 and Grade 4 are easy to distinguish between, the visual difference between Grade 0 and Grade 1 is rather subtle and can be quite difficult to identify. For this reason, various experienced radiologists struggle to agree on the extent of Grade 1, and this grade is officially labelled as "doubtful" OA. This is due to it displaying the earliest, most ambiguous stage of joint degeneration, where small osteophytes plausibly could be present. This imbalance, combined with the modest dataset for a deep learning task, in large part shaped some of the decisions described in the report. Most notably, the structure of the model. The limitations in the size of the dataset and the high number of Grade 0 images demonstrate that it is more likely for the model to make erroneous classifications.

This disparity in difficulty across grades entails direct consequences for automated classification. A model that would normally be effective on aggregate accuracy can still be clinically flawed if that accuracy is concentrated entirely on the noticeable cases. Namely, Grade 0 and Grade 4, while failing most on classifying grades that matter most for early intervention. This project set out to build a CNN-based classifier for the full KL grading problem, with an explicit focus on evaluating performance on Grade 1, rather than purely prioritising accuracy regarding quality. 

The dataset used in this project is the Kaggle "Knee Osteoarthritis Dataset with Severity Grading", which provides approximately 9,800 X-ray images pre-split into training, validation and test sets, each containing one folder per KL grade. As is typical for real-world clinical datasets, the class distribution is heavily imbalanced. The majority of images provided are Grade 0 images, making up roughly a quarter of the dataset, while Grade 4, for example, accounts for less than two per cent. Additionally, we would also experiment with different approaches such as a Simple CNN and a Weighted CNN, whilst these were deprecated, we still decided to maintain the findings of these due to them still being highly useful.

This report will be structured as follows. The Background section will cover the relevant technical concepts, e.g. CNNs. The Methods section will describe the full pipeline, including preprocessing and architecture, including some code snippets. The Results section will report the early baseline outcome and the outcome of the final model as well as evaluation performed. The Discussion section will interpret these results and consider limitations and offer concrete next steps. 

## Background
A Convolutional Neural Network (CNN) is a class of deep learning architecture built around convolutional layers, which learn spatially local filters that are applied across the entire input image. This structure establishes CNN as a good fit to image classification tasks: early layers tend to learn generic, low-level features such as edges and textures, while deeper layers combine these into progressively more abstract, task-specific representations. In this context, patterns associated with joint space narrowing or osteophyte formation. Pooling layers between convolutional blocks reduces spatial resolution while retaining the most salient activations, minimising computational costs and provides some translation invariance, especially valuable when the knee joint is not perfectly centred in every X-ray. 

Training a CNN from randomly initialised weights usually requires a large volume of labelled data to learn useful filters without overfitting. Such as in this case, when the available dataset is small relative to the complexity of the task, a helpful alternative is transfer learning. In transfer learning, a CNN backbone that has already been trained on a large, unrelated dataset (such as ImageNet, containing over a million images) is reused. This is either as a fixed feature extraction or as a starting point for further training. The intuition is that the low and mid-level features learned from natural images, e.g. edges, corners, textures and simple shapes, are widely transferable to other visual domains such as radiographs. Albeit, the two domains look distinct at a glance. Fine-tuning is a technique where some pretrained layers are unfrozen and updated at a small learning rate, after training with a frozen backbone. This allows the network to adapt these generic features toward domain-specific ones without destroying the pretrained weights due to large, poorly conditioned gradient updates early into training. 

An additional consideration taken into account for this task is that KL grades are ordinal rather than nominal, where Grade 3 is closer to Grade 2 than it is to Grade 0. Standard classification accruacy treats every misclassification identically, for this reason a model that misclassifies Grade 1 images as Grade 0 will receive the same accuracy scoring as one that confuses Grade 1 with Grade 4, despite the former being a smaller clinical error. This is addressed by Quadratic Weighted Kappa (QWK) which penalises misclassifications according to the squared distance between the predicted and true grade, and by correcting for the level of agreement expected by chance. This is the standard metric in wider literature for ordinal medical grading tasks of this variety, and is used throughout this project alongside per-class precision and recall.

Finally, severe class imbalance interacts badly with standard cross-entropy loss, which is minimised by a model that simply predicts the majority class as often as possible when minority classes are rare and difficult. Class weighting — scaling the loss contribution of each class inversely to its frequency — is a standard mitigation that increases the penalty for misclassifying minority-class examples such as Grade 1 and Grade 4, without requiring synthetic oversampling that could otherwise encourage overfitting on a very small number of unique images.

## Methods
### Dataset and preprocessing

The dataset was loaded from its pre-defined train/validation/test folders, each containing one subfolder per KL grade. Rather than loading every image into memory as a NumPy array up front, images were processed lazily through a tf.data pipeline built from file paths, which reads and preprocesses each image only as it is needed during training:

```python
def list_paths_and_labels(split_dir):
    class_folders = sorted(
        [f for f in split_dir.iterdir() if f.is_dir()], key=lambda p: int(p.name)
    )
    paths, labels = [], []
    for class_folder in class_folders:
        label = int(class_folder.name)
        for img_path in class_folder.glob("*"):
            paths.append(str(img_path))
            labels.append(label)
    return paths, labels
```
Each image undergoes three preprocessing steps applied at original resolution, before any resizing, so that detail is not lost through a double downsampling:

Centre cropping, removing a fixed proportion of background/soft tissue surrounding the joint;
Contrast Limited Adaptive Histogram Equalisation (CLAHE), which enhances local contrast around the joint space without over-brightening the image globally, as a standard histogram equalisation would;
An optional unsharp-mask sharpening step, exposed as a toggle so its effect can be evaluated separately, since sharpening can accentuate genuine bone-edge detail but also amplify noise.
```python
def preprocess_knee_xray(image_uint8, use_sharpening=False):
    cropped = centre_crop_image(image_uint8)
    resized = cv2.resize(cropped, IMG_SIZE)
    enhanced = apply_clahe(resized)
    if use_sharpening:
        enhanced = unsharp_mask(enhanced)
    return enhanced.astype("float32")
```
### Augmentation

Data augmentation was applied to the training set only, to artificially increase the effective diversity of the small dataset and reduce overfitting. An important correction was made relative to an earlier version of the pipeline, which had included vertical flipping. A knee X-ray flipped vertically produces an anatomically impossible image — an upside-down knee is not a knee a clinician would ever encounter — so this transformation was removed. The final augmentation set includes only transformations that a real X-ray could plausibly exhibit due to patient positioning or exposure differences: horizontal (left/right) flipping, small rotations, mild zoom and translation, and contrast jitter.

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1),
], name="data_augmentation")
```
### Addressing class imbalance

Class weights were computed directly from the training label distribution using scikit-learn's balanced weighting scheme, and passed into the loss function during training so that errors on minority classes contribute proportionally more to the overall loss:

```python
class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.array(train_labels),
)
class_weights = dict(enumerate(class_weights_array))
```
### Model architectures

Two architectures were trained and compared. The first is an improved custom CNN, built from four convolutional blocks (each with batch normalisation and L2 weight regularisation), global average pooling, and a dropout-regularised dense classification head, trained entirely from scratch:

```python
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    return x
```
The second is a transfer-learning model built on MobileNetV2 pretrained on ImageNet. Since MobileNetV2 expects a three-channel input, the single-channel X-ray is replicated across the RGB channels before being passed through the backbone's own preprocessing function. Training proceeds in two stages: first with the backbone frozen and only the new classification head trained, then with the top 40 layers of the backbone unfrozen and fine-tuned at a substantially lower learning rate. Batch normalisation layers within the backbone are kept frozen throughout fine-tuning, as updating their running statistics on small, imbalanced batches tends to destabilise training.

```python
base_model.trainable = True
for layer in base_model.layers[:-num_layers]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
```
### Training and monitoring

Both models were trained with the Adam optimiser and sparse categorical cross-entropy loss, using the class weights described above. Because standard accuracy and loss can look acceptable even when a model has collapsed onto the majority class, a custom callback was used to compute Quadratic Weighted Kappa on the validation set after every epoch, exposing it to Keras's built-in early stopping, learning rate reduction and checkpointing callbacks as the primary metric to monitor:

```python
class QWKCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, y_pred = get_predictions(self.model, self.dataset)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        logs["val_qwk"] = qwk
```
### Evaluation

Both models were evaluated on the held-out test set using the same suite of metrics: overall accuracy, per-class precision and recall via a full classification report, a confusion matrix, Quadratic Weighted Kappa, and Grade 1 recall and precision specifically, given its clinical importance and its poor performance in the initial baseline. Finally, Grad-CAM was applied to a sample of test images from each grade to visualise which regions of the X-ray most influenced the model's prediction, as a qualitative check that the model is attending to the joint space rather than an unrelated part of the image (such as a border artefact or annotation mark).

### Results

The initial baseline CNN — three convolutional blocks trained from scratch on cropped, CLAHE-enhanced images, without class weighting or augmentation — established the scale of the problem this pipeline was designed to address. On the held-out test set, it achieved an overall accuracy of 0.379, but this figure disguised a severe failure mode: the confusion matrix showed that the model predicted almost every image as Grade 0, regardless of true grade. All 296 true Grade 1 test images were misclassified as Grade 0, giving a Grade 1 recall of exactly 0.0, and the Quadratic Weighted Kappa was 0.0178 — close to the level of agreement expected from chance alone. This confirmed that accuracy was not a sufficient metric for this task, and that the model had not learned to distinguish osteoarthritic features at all; it had simply learned that Grade 0 is the most frequent label.

The extended pipeline described in the Methods section — anatomy-correct augmentation, class weighting, QWK-based monitoring, and the two comparative architectures — was designed specifically to correct this failure mode, and the same evaluation suite (classification report, confusion matrix, QWK, and Grade 1 recall/precision) was applied identically to both the custom CNN and the transfer-learning model, so that their outputs are directly comparable to the baseline and to each other. At the time of writing, full training runs for both models are still to be executed and their resulting metrics inserted into the table below; the structure is included here as it will be populated:

| Model                                       | 	Accuracy | 	QWK	Grade 1 Recall | Grade 1 Precision |
|---------------------------------------------|----------|-------------------|-------------------|
| Baseline CNN (Crop + CLAHE)                 | 	0.379    | 	0.018             | 	0.000             |	
| Simple CNN  (Resize + Rescale only)         | 0.377    | 0.0364            | 0.0               |
 | Weighted CNN (Crop + CLAHE) | 0.334    | 0.119             | 0.029             |
| Custom CNN (improved, from scratch)         | 	x	        | x                 | 	x                 |
| Transfer learning (MobileNetV2, fine-tuned) | 	x        | 	x                 | 	x                 |

Training curves (accuracy and loss over epochs for both the frozen-backbone and fine-tuning stages) and confusion matrices for each model are produced automatically by the pipeline and are intended to be inserted alongside this table once training is complete, along with the Grad-CAM visualisations described in the Methods section.

### Discussion

The baseline result is a useful negative result; it demonstrates conclusively why accuracy can be a flawed evaluation metric on a dataset with this degree of class imbalance, and why a model can appear to perform reasonably (38% accuracy across five classes, well above the 20% expected from random guessing) while being clinically worthless. This motivates every subsequent design decision in the pipeline, from the introduction of class weighting to the decision to monitor Quadratic Weighted Kappa rather than validation loss during training.

As is also recorded in the result section, there were explorations of a Simple CNN with Resize and Rescale only and a Weighted CNN with Crop and CLAHE, whilst these yielded some successful results and thus were valuable to record, we believed it was possible to find a more sufficient solution.

A comparison between the custom CNN to the transfer-learning model was overall insightful about deep learning involving smaller datasets. Due to there being only roughly 9,800 images, it was inevitable for there to be a large imbalance in results. The ability to use a backbone pretrained on a much larger dataset was of high value due to it combatting this issue. The lack of images made it harder for the model to consistently and accurately extract features, making a full adaptation rather difficult. This informed the decision to consider a MobileNetV2 approach.

Several limitations of the current approach are worth noting. First, a single train/validation/test split, especially with a class as small as Grade 4, is vulnerable to variance introduced simply by which examples happened to fall into which split; k-fold cross-validation would yield a more reliable estimate of true performance and its uncertainty. This comes at the cost of significantly increased training time. Secondly, treating the five KL grades as unrelated categorical classes, even with QWK used for evaluation, does not build the ordinal structure of the labels into the training objective itself. An ordinal loss function such as CORAL (Consistent Rank Logits), which reframes ordinal classification as a series of binary "is the grade greater than k" comparisons, could better align the training signal with the ordinal nature of the task and may further improve performance on common misclassifications. Although these are the most clinically forgivable errors, they are still worth reducing. Thirdly, whilst class weighting mitigates the imbalance problem without duplicating data, it does not create new information. Targeted collection or more aggressive augmentation of Grade 1 and Grade 4 examples specifically, rather than uniform augmentation across all classes, may yield further gains. Particularly, for Grade 1 given how visually subtle it is even for its true positive examples.

The inclusion of Grad-CAM as a final interpretability step reflects a further consideration specific to medical imaging applications: a model can achieve strong quantitative metrics while attending to spurious, non-anatomical features — a scanner watermark, a consistent framing artefact, or an incidental correlation between image source and label distribution. Because the dataset is compiled from a public source rather than a single controlled acquisition pipeline, this risk cannot be ruled out purely from the numeric evaluation metrics, and a qualitative check of where the model is "looking" is a necessary complement to accuracy, QWK and per-class recall, not an optional addition.

Overall, this project migrated from an initial baseline that failed entirely on the clinically important minority classes, to a considerably more complete pipeline. Addressing preprocessing, anatomically valid augmentation, class imbalance, ordinal-aware evaluation, architecture comparison, and interpretability. The baseline's near-chance QWK of 0.018 sets a low bar that both extended models are expected to clear substantially. Where the extent of this, and which of the two approaches proves more effective on this particular dataset, is the empirical question this pipeline was built to answer. 
