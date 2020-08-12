# InsightFace_Pytorch
    Dataset:
        Train: deeplign, celebA
        Test: facescrub, lfw
    model:
        backbone: resnet50/100/152
        head: Arcface, CurricularFace
##DONE
    research paper: ArcFace, Face Recognition Survey
    preprocess all dataset.( datapine.py, preprocess.py, celebA_preprocess.py)
    customize evaluate( evaluate_custom() in Learner.py)
    test model with their weight.
##TODO
    train:
        model: resnet50 + arcface
        dataset: celebA or deeplign
