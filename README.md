# InsightFace_Pytorch

> - Dataset:
>   - Train: deeplign, celebA
>   - Test: facescrub, lfw
> - model:
>   - backbone: resnet50/100/152
>   - head: Arcface, CurricularFace

## DONE

- research paper: ArcFace, Face Recognition Survey
- preprocess all dataset.( datapine.py, preprocess.py, celebA_preprocess.py)
- customize evaluate( evaluate_custom() in Learner.py)
- test model with their weight.
- evaluate (only inter) pre-trained model with test dataset (identity card) (1)
- augment test dataset -> evaluate intra (simple level)

## TODO

- research paper:
  - image pre-processing for face recognition
  - post processing for face recognition
  - data augment face
  - face recignition system
- finetuning their pre-trained resnet50
- train model resnet100 with MS1M, deepGlint

### **Face Detection**

1. Survey face detection:

- Understand module, code. run repository successfully with their pre-trained
  model.
- Test their model on image, video to estimate paremeters,runtime about model.
  some model you can test:

  1.  [RentinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
  2.  [mtcnn](https://github.com/anhdhbn/facerec/tree/master/mtcnn_pytorch)
  3.  [yolov3](https://github.com/ultralytics/yolov3)
  4.  opencv, dlib

  > You can survey other models about face detection from github, paper, ..etc.

2. Train model yourself

- dataset: survey face dataset or dataset in repository.
- train: preprocess data and fit them to module to train.
- evaluate own model.

3. develop model come face detection system.
