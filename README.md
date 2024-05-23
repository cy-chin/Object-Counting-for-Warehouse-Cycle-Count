<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# DSI Capstone Project - Object-Counting for Warehouse Cycle Count - Part 1: A Machine Learning Approach

Authors: ChungYau Chin

-[Executive Summary](#executive-summary)
-[Problem Statement](#problem-statement)
-[Data Source](#data-source)
-[Data Dictionary](#data-dictionary)
-[Data Challenges](#data-challenges)
-[Exploratory Data Analysis](#exploratory-data-analysis-eda)
-[Data Preprocessing](#data-preprocessing)
-[Machine Learning Model](#machine-learning-model)
-[Model Evaluation](#model-evaluation)
-[Model Training](#model-training)
-[Conclusion](#conclusion)


---

## Executive Summary

Cycle count (also known as "Stock Take") is a critical inventory management process that ensures physical inventories in the warehouse are aligned with records in the system. Consistent and accurate inventory levels are key to timely order fulfillment and smooth warehouse operations.

There are different methods of Cycle Count such as ABC Count, Usage-based Count, Opportunity-based Count, Zone Counting, and Hybrid Count [^l1]. Regardless of the methods, there are multiple sequential actions taking place in every Cycle Count, as depicted in Figure 1. While some of the steps have benefited from digitization and automation, Step 3 (Count) and Step 4 (Reconcile) remains a labor-intensive process in many warehouse operations.

![Cycle-Count-Steps](./images/cycle-of-cycle-counting.png)
*Figure 1: The Cycle of Cycle Counting

In this project, I'm excited to embark on a quest to explore how machine learning can be adopted to assist, or even potentially replace, the manual inventory counting process. Using the Amazon Bin Image Dataset[^l3], I explored the various pre-trained Convolutionary Neural Network (CNN) models, as well as a customized CNN, in an attempt to established a CNN model that is able to count the number of objects in a bin. From the model evaluation, DenseNet stood up for its high overall accuracy XXXX, as well as relatively well perform in the Recall in each class. Hyperparameter tuning on the final model has resulted in overall accuracy XXX. This project demonstrated the feasibility of using CNN for object counting, setting stage for counting automation even without high quality camera setup.

## Problem Statement

The reliance on manual cycle counts in warehouse has greatly limited the scalability of warehouse operations and affects overall productivity.

How can machine learning enable inventory counting automation, providing accurate, fatigue-free results that increase productivity and reduce labor reliance in warehouse operations?

## Data Source

The Amazon Bin Image Dataset[^l3] will be used as the dataset to train the machine learning model. The dataset contains over 500,000 images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset were captured as robot units carry pods as part of normal Amazon Fulfillment Center operations.

## Data Dictionary

Each image comes with a JSON message that serves as the metadata to describe the images.
Figure 2 shows a sample of the JSON message for `00001.jpg` image. Attribute `BIN_FCSKU_DATA` carries the SKUs (Stock Keeping Unit, i.e. product model) available in the bin. The SKU details, such as description, quantity, dimension, etc is provided in the Level -2 to -4 attributes. Attribute `EXPECTED_QUANTITY` shows how many objects available in the bin. This will be used as the label for the bin.

```
{
  "BIN_FCSKU_DATA": {
    "B000C33MI2": {
      "asin": "B000C33MI2",
      "height": {
        "unit": "IN",
        "value": 2.79921259557
      },
      "length": {
        "unit": "IN",
        "value": 3.90157479917
       },
      "name": "FRAM XG7317 ULTRA Spin-On Oil Filter with Sure Grip",
      "normalizedName": "FRAM XG7317 ULTRA Spin-On Oil Filter with Sure Grip",
      "quantity": 4,
      "weight": {
        "unit": "pounds",
        "value": 0.3000050461296
      },
      "width": {
        "unit": "IN",
        "value": 2.90157480019
      }
    },
    "B0050Z27KG": {
      "asin": "B0050Z27KG",
      "height": {
        "unit": "IN",
        "value": 0.899999999082
      },
      "length": { 
        "unit": "IN",
        "value": 11.299999988474
      },
      "name": "Suncatcher - Axicon Rainbow Window - Includes Bonus \"Rainbow on Board\" Sun Catcher",
      "normalizedName": "Suncatcher - Axicon Rainbow Window - Includes Bonus \"Rainbow on Board\" Sun Catcher",
      "quantity": 2,
      "weight": {
        "unit": "pounds",
        "value": 0.5
      },
      "width": {
        "unit": "IN",
        "value": 7.699999992146
      }
    },
    "B01BV89HNU": {
      "asin": "B01BV89HNU",
      "height": {
        "unit": "IN",
        "value": 2.1999999977560005
      },
      "length": {
        "unit": "IN",
        "value": 3.99999999592
      },
     "name": "Type C, iOrange-E 2 Pack 6.6 Ft Braided Cable for Nexus 6P, ... Nokia N1   Tablet and Other USB C Devices, Black",
     "normalizedName": "2 Pack Type C, iOrange-E 6.6 Ft Braided Cable for ... Nokia N1 Tablet and Other USB C Devices, Black",
     "quantity": 6,
     "weight": {
       "unit": "pounds",
       "value": 0.3499999997064932
     },
     "width": {
       "unit": "IN",
       "value": 3.899999996022
     }
   }
 },
 "EXPECTED_QUANTITY": 12
 }
```
Figure 2: JSON metadata of 00001.jpg

## Data Challenges

***Assumptions***

- The metadata gave accurate information about the items and quantity in the Bin.

***Limitation***

- The photos were taken in low resolution camera, where each jpeg file is only around 10Kb in size. The photos are noisy, obfuscated and low quality.
- The overall 536,434 images, with total size 3.34GB, poses a challenge in computational resources


Each bin has a mixed number of SKUs. There are huge set of SKUs, therefore object detection with bounding boxes as the image annotation is not practical. The counting of objects across mixed of SKUs, that is the challenge.  

***Approach***

- Alignining with the moderate difficulty level in the Amazon Bin Image Dataset (ABID) Challenge[^l4], only the bin images that contain up to 5 objects will be counted.  

## Exploratory Data Analysis (EDA)

- Total 536,434 images. One image is the photo taken for 1 Bin.

![Amazon Bin Images](./images/AmazonBinImages.png)

Figure 3: Snapshot of the Bin Images

- Based on the Bin Item Qty Histogram, two-third (67.48%) of the images were taken for bins with Qty 5 or below (including empty bin). Beyond Qty 5, the incremental 

![Bin Item Quantity Histogram](./images/BinItemQtyHistogram.png)

|Bin Item Quantity | Number of Images|Percentage of Images |Cumm. Sum.|
| --- | ---|--- |---|
|0|9901|1.85% |1.85%|
|1|41347|7.71% |9.55%|
|2|77063|14.37%|23.92%|
|3|90258|16.83%|40.74%|
|4|80750|15.05%|55.80%|
|5|62648|11.68%|67.48%|
|6|46058|8.59%|76.06%|
|7|33684|6.28%|82.34%|
|8|24219|4.51%|86.86%|
|9|17298|3.22%|90.08%|
|10|12823|2.39%|94.47%|
|...|...|...|...|

*In this project, due to limited time and resources, the machine learning will attempt to count up to 5 quantities in a Bin, which will cover two-third (67.48%) of the total available images. Total 361,967 images will be used for model training*

## Data Preprocessing

There were two parts in the data preprocessing:

1. Item and quantity information were extracted from the metadata into a inventory dataframe in order to facilitate image filtering/selection for the model training and evaluation.
2. Image was resized (to 224x224x3) and converted to numpy ndarray format. The associated label (i.e. expected quantity count in the bin) was extracted from the inventory dataframe to form the label data in the machine learning training.

Note - Image Augmentation will be used during model training, through configuration using Keras ImageDataGenerator as a data generator during model fit. Therefore, there is no additional coding required for Image Augmentation.

## Machine Learning Model

The following deep learning models were explored. Transfer learning was employed by utilizing pre-trained convolutional neural networks (CNNs) trained on the ImageNet dataset. The original classification layers in these CNNs were removed and replaced with new, fully connected layers. These modified networks were then trained to classify the counted quantity in Amazon Bin Images.

| Model|
| --- |
| MobilenetV2 |  
| EfficientNetV2M |  
| DenseNet|
| VGG19|
| InceptionResNetV2|
| ResNet152V2|

Image Augmentation technique was applied through the use of ImageDataGenerator by Keras.

## Model Evaluation

| Model|Accuracy|
| --- | ---|
| MobilenetV2 |0.3963|
| EfficientNetV2M |0.1642|
| DenseNet|0.4375|
| VGG19|0.3979|
| InceptionResNetV2|0.4142|
| ResNet152V2|0.4187|

DenseNet201 has the best performance among the models evaluated.

## Model Training

Pretrained model of DenseNet201 on ImageNet dataset was used for the final model training, using 361,967 images for bins with quantity 0 to quantity 5. Total 114,057,990 parameters were trained and updated. Overall accuracy of the trained model was 0.441.

## Conclusion

Using machine learning to count object in a bin is feasible approach. The performance can be greatly enhance if the quality of the images are in better conditions. Transfer learning has enhanced the training process and result, comparing to training a model from scratch. Given more time and resource, other techniques like parallel modeling and voting ensembles can be explored for better performance.


*Reference:*

[^l1]: https://www.bigcommerce.com/glossary/inventory-cycle-count/
[^l2]: https://www.netsuite.com.sg/portal/sg/resource/articles/inventory-management/using-inventory-control-software-for-cycle-counting.shtml
[^l3]: https://registry.opendata.aws/amazon-bin-imagery
[^l4]: https://github.com/silverbottlep/abid_challenge
