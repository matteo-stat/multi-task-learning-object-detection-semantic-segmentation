# Simultaneous Lightweight Object Detection and Semantic Segmentation (SSD and DeepLabV3)

## Welcome!

Hey there! 🎉 This repo is all about a fun and challenging project I worked on for a computer vision university exam during 2023. The goal was to build a network that can handle both object detection and semantic segmentation at the same time. I’ve put together a lightweight solution that could be great for real-time mobile apps.

### The Goal

I wanted to create a network that could do two things:
1. **Object Detection**: Detect and label objects in images using SSD (Single Shot MultiBox Detector).
2. **Semantic Segmentation**: Identify and segment different parts of an image using DeepLabV3+.

To keep things light and fast, I used MobileNetV2 and even tried ShuffleNetV2 (though I didn’t had time to train and test it, I hope it works). I've used deptwhise separable convolutions when implementing SSD, so technically I've implemented what's called SSDLite. However it's exactly the same as SSD, just switching standard convolutions operations with depthwise separable convolutions.

### Key Highlights

- **Built from Scratch**: Everything, including the backbone networks and detection/segmentation heads, were coded from scratch using NumPy and TensorFlow.
- **Compact and Efficient**: The final network has around 4M parameters, which is pretty small and efficient for real-time tasks.
- **Mobile Architectures**: Implemented MobileNetV2 and ShuffleNetV2 layer by layer from scratch (as said above I tested only MobileNetV2, hope ShuffleNetV2 works too).
- **Training**: Trained the models on a proprietary dataset from my company with three classes: people, forklifts, and rails.

### Results

The models didn’t break any records, but considering the small dataset and the fact that everything was built and trained from scratch, the results are pretty cool! It shows that even with limited data, these computer vision techniques can be quite effective.

### What I Learned

This project was.. challenging! I learned a lot about building and training models from scratch and got a better understanding of how to make efficient, real-time computer vision solutions. It wasn't easy but seeing decent boxes and segmentation masks coming out from the network was super rewarding.

## Why and this Repo and What’s inside

I hope this repo can help other people getting to know easier architectures for performing object detection and semantic segmentation

I've implemented everything using Numpy and Tensorflow only, organizing the code in a custom python model called **`ssdseglib`**, where each functions it's document with type hints and docstrings. Unfortunately I wrote all the code with the time constraints of university exams while working full time. I've found the right time to comment stuff better only 1 year later and I've noticed that a lot of stuff could have been done easier and simpler, especially without using custom classes. But apart from that I think all the functions and steps are quite documented and straightforward, so I hope it will be useful to someone else.

Here's a quick breakdown of the repo: 

- **`/data`**: this folder contains metadata for training, validation and test, but unfortunately the dataset it's proprietary and I cannot share it outside my company.
- **`/models`**: this folder contains trained model, I shared MobileNetV2 with SSDLite and DeepLabV3+ trained for 105 epochs.
- **`/ssdseglib`**: this is a custom python module were all the code for this project it's organized, the naming of the files should be self-explanatory and there are type hints, docstrings and comments for helping having an easier understanding.
- **`01-ssd-framework-single-shot-detector-for-object-detection.ipynb`**: in this notebook I explain the core idea behind SSD framework for object detection.
- **`02-data-encoding-and-decoding.ipynb`**: in this notebook I explain the data formats expected by networks using SSD for object detection.
- **`03-multi-task-network-ssdlite-deeplabv3plus-training.ipynb`**: in this notebook I run the main experiment, so you'll find data loading, model training and results.
- **`99-check-dataset-class-imbalance`**: ignore it, still there for me, I've used it to calculate some class weights for balancing the training.

## Getting Started

1. Clone the repo.
2. Install the necessary packages using the `requirements.txt` file.
3. Customize and run your experiments.

Feel free to explore, give feedback, or contribute if you’d like!

## Conclusion

This project was a great learning experience and a fun challenge. It’s a good example of how you can handle object detection and segmentation together with a small, efficient network. Hope you find it interesting too!
