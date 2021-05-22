# Train with BIWI(70%), test with BIWI(30%)

|Method|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|
|DeepHeadPose|5.67|5.18|-|-|
|FSA-Net|2.89|4.29|3.6|2.6|
|VGG16|3.91|4.03|3.03|3.66|
|Multitask-NetV1(normal)|4.53|4.6|3.47|4.2|
|Multitask-NetV1(tanh function)|5.49|3.92|3.21|4.21|
|Multitask-NetV2(euler angle)|6.02|5.33|5.11|5.48|
|**Multitask-NetV2(vector base)**|**5.33**|**3.9**|**3.28**|**4.17**|

</br></br></br>
# Train with 300WLP, test with BIWI

|Method|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|
|FAN|8.53|7.48|7.63|7.89|
|Hopenet(a=2)|5.17|6.98|3.39|5.18|
|Hopenet(a=1)|4.81|6.61|3.27|4.9|
|SSR-Net-MD|4.49|6.31|3.61|4.65|
|FSA-Net|4.27|4.96|2.76|4.00|
|Multitask-NetV2(euler angle)|4.64|7.23|6.23|6.03|
|**Multitask-NetV2(vector base)**|**4.62**|**3.29**|**4.52**|**4.14**|

</br></br></br>
# Train with CMUDataset(70%), test with CMUDataset(30%)

|Method|Iou|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Multitask-NetV2(euler angle)||||||
|Multitask-NetV2(vector base)||||||

</br></br></br>
# All our methods

|Method|Iou|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Multitask-NetV1(normal)|63.8|4.53|4.6|3.47|4.2|
|Multitask-NetV1(tanh function)|73.4|5.49|3.92|3.21|4.21|
|Multitask-NetV2(euler angle)|60.72|6.02|5.33|5.11|5.48|
|**Multitask-NetV2(vector base)**|**63.6**|**5.33**|**3.9**|**3.28**|**4.17**|

