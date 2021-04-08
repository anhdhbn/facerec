# Train with BIWI(70%), test with BIWI(30%)

|Method|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|
|DeepHeadPose|5.67|5.18|-|-|
|FSA-Net|2.89|4.29|3.6|2.6|
|VGG16|3.91|4.03|3.03|3.66|
|Multitask-NetV1(normal)|4.53|4.6|3.47|4.2|
|Multitask-NetV1(tanh function)|5.49|3.92|3.21|4.21|
|Multitask-NetV2(euler angle)|6.02|5.33|5.11|5.48|
**|Multitask-NetV2(vector base)|5.33|3.9|3.28|4.17|**

</br></br></br>
# Train with 300WLP, test with BIWI

|Method|Yaw|Pitch|Roll|MAE|
|:-:|:-:|:-:|:-:|:-:|
|Hopenet(a=2)|6.47|6.56|5.44|6.16|
|Hopenet(a=1)|6.92|6.64|5.67|6.41|
|SSR-Net-MD|5.14|7.09|5.89|6.01|
|FSA-Net|4.5|6.08|4.64|5.07|
|Multitask-NetV2(euler angle)|||||
|Multitask-NetV2(vector base)|7.62|6.29|4.52|6.21|

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
|Multitask-NetV2(vector base)|63.6|5.33|3.9|3.28|4.17|

