# Dog Breed Classifier

A project from my Udacity AI Programming with Python course. It uses a pretrained convolutional neural network (CNN) to:

1. **Identify** which pet images are of dogs and which aren't.
2. **Classify** the breed of dog, for the images that are of dogs.

CNNs work particularly well for detecting features in images — colors, textures, edges — and using them to identify objects. This project uses a CNN already trained on [ImageNet](https://www.image-net.org/), a dataset of 1.2 million images, and compares three architectures — **AlexNet**, **VGG**, and **ResNet** — to determine which performs best for this task.

## Setup

```bash
git clone https://github.com/phuonganh0804/dog-breed-classifier.git
cd dog-breed-classifier

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run all three CNN architectures against the sample images in `pet_images/`:

```bash
sh run_models_batch.sh
```

This classifies 40 pet images per model and writes each model's results to a `.txt` file.

To classify your own images: add them to `uploaded_images/`, then run:

```bash
sh run_models_batch_uploaded.sh
```

This runs `check_images.py` with all three architectures against the images in `uploaded_images/` and writes their results files to the project root.

## Results

Results from running `check_images.py` with each of the three CNN architectures on the 40 sample pet images (30 dog images, 10 non-dog images):

| Metric | AlexNet | ResNet | VGG |
|---|---|---|---|
| Label match (%) | 75.0 | 82.5 | 87.5 |
| Correctly classified as dog/not-dog (%) | 100.0 | 100.0 | 100.0 |
| Correctly classified breed (%) | 80.0 | 90.0 | 93.3 |
| Correctly classified not-dog (%) | 100.0 | 90.0 | 100.0 |

**Objective 1** (dog vs. not-dog): both VGG and AlexNet correctly classify images as "dog" or "not-a-dog" 100% of the time; ResNet reaches 90%.

**Objective 2** (breed classification): VGG performs best, correctly identifying the dog breed over 90% of the time.

**Conclusion**: VGG is the best-performing architecture overall — it's tied for the top score on dog/not-dog classification (100%) and has the highest breed classification accuracy (93.3%). ResNet classifies breeds more accurately than AlexNet, but only VGG and AlexNet achieve 100% accuracy on dog/not-dog classification.
