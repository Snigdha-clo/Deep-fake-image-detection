# DeepFake Image Detection using Vision Transformer (ViT-B/16)


A deep learning mini-project that classifies images as Fake or Real using a fine-tuned Vision Transformer (ViT-B/16).
Project Overview
Deepfake images can be difficult to distinguish from authentic images through visual inspection alone. This project explores transfer learning with Google's ViT-B/16 architecture to identify visual patterns that help distinguish fake images from real ones.

The project uses the OpenDeepfake-Preview dataset and the official Google Research Vision Transformer implementation built with JAX and Flax.
Model and Approach
- Architecture: Vision Transformer Base/16 (ViT-B/16)
- Pretrained weights: ImageNet-21K checkpoint
- Framework: JAX and Flax
- Task: Binary image classification (Fake / Real)
- Input size: 224 × 224 RGB images
- Optimizer: AdamW
- Learning rate: 1e-5
- Batch size: 2
- Fine-tuning epochs: 1 (the final recorded experiment)
- Weight decay: 1e-4
Workflow
1. Load the OpenDeepfake-Preview dataset.
2. Create stratified training, validation, and test splits (70%, 15%, and 15%).
3. Convert images to RGB, resize them to 224 × 224, and normalize pixel values to the range [-1, 1].
4. Load the pretrained ViT-B/16 model and adapt its classification head for two classes.
5. Extract frozen ViT features and train a Logistic Regression baseline. Select its regularization parameter using validation ROC-AUC.
6. Fine-tune the ViT-B/16 model using AdamW and the training split. Use the validation split to monitor performance and save the best checkpoint.
7. Evaluate the fine-tuned model on the held-out test split.
8. Generate a confusion matrix and ROC curve.



