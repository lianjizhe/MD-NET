Modality-aware Distillation Network for Microvascular Invasion Prediction of Hepatocellar Carcinoma from MRI Images
## I. Project Overview
This project mainly implements a medical image classification model using knowledge distillation. It combines a student model and a teacher model for training. The model processes medical image data related to HCC (Hepatocellular Carcinoma) and utilizes clinical features as auxiliary information to improve the accuracy of classification.
## II. Code Structure
### 1. main.py
Function: The main program of the entire project. It is responsible for setting random seeds, configuring parameters, initializing models, defining loss functions and optimizers, as well as training and validating the models.
### 2. dataset.py
Function: Defines the dataset class hcc_3d_all for loading and processing HCC - related medical image data, including data reading, pre - processing, and augmentation.
### 3. pytorch/unet_student.py
Function: Defines the student model, including modules such as targetNet and UNet3D, for feature extraction and classification.
### 4. pytorch/unet_teacher.py
Function: Defines the teacher model, including modules such as targetNetc and UNet3D, similar to the student model but incorporating clinical features during processing.
### 5. enet_model.py
Function: Defines the L2 loss function class L2Loss for calculating the model's loss.
### 6. pytorch/transformer.py
Function: Defines Transformer - related modules, including Embeddings, Attention, Mlp, etc., for processing image features.
### 7. pytorch/unet_concat.py
Function: Defines the combined models of the student and the teacher, including concat_studentNet and concat_teacherNet, which concatenate and process the outputs of multiple models.
Key Parts:
concat_studentNet class: The combined student model, which concatenates and processes the outputs of the pre - model and the HBP model.
concat_teacherNet class: The combined teacher model, which concatenates and processes the outputs of the pre - model and the HBP model.
## III. Usage
### 1. Configure Parameters
In main.py, you can configure the relevant parameters of the experiment through command - line arguments. For example:
```bash
python main.py --data_dir /home/hcc/ --label_dir ./label --ckpt_dir ./hcc_model/ --sequence arterial --gpu 1 --batch_size 16 --lr 3e-4
```
### 2. Train the Model
Run main.py to train the model:
```bash
python main.py
```
### 3. Validate the Model
The eval_training function implements the model validation process, which can be called during the training process for validation.

## IV. Notes
Please ensure that the paths of the dataset and the label file are correct.
You can adjust the hyperparameters of the model, such as the learning rate and batch size, as needed.
The random seeds in the code have been set to ensure the reproducibility of experiments, but there may be slight differences in different environments.
