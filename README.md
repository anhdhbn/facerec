# InsightFace_Pytorch

>- Dataset:
>   - Train: deeplign, celebA
>   - Test: facescrub, lfw
>- model:
>   - backbone: resnet50/100/152
>   - head: Arcface, CurricularFace

## DONE

- research paper: ArcFace, Face Recognition Survey
- preprocess all dataset.( datapine.py, preprocess.py, celebA_preprocess.py)
- customize evaluate( evaluate_custom() in Learner.py)
- test model with their weight.
- evaluate (only inter) pre-trained model with test dataset (identity card)    (1)
- augment test dataset -> evaluate intra (simple level)

## TODO

- research paper:
  - image pre-processing for face recognition
  - post processing for face recognition
  - data augment face
  - face recignition system
- finetuning their pre-trained resnet50
- train model resnet100 with MS1M, deepGlint

