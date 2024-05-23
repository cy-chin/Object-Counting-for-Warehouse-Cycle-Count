<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# DSI Capstone Project - Automated Inventory Counting in Fulfillment Centers Using Deep Learning

Authors: ChungYau Chin

- [Executive Summary](#executive-summary)
- [Data Source](#data-source)
- [Challenges and Assumptions](#challenges-and-assumptions)
- [Approach](#approach)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Result and Future Work](#results-and-future-work)
- [Conclusion](#conclusion)


---

## Executive Summary

Cycle counting (also known as stocktaking) is a critical inventory management process ensuring that physical inventory in a warehouse matches recorded quantities. Maintaining accurate inventory levels is crucial for efficient order fulfillment and smooth warehouse operations. While various cycle counting methods exist (e.g., ABC counting, zone counting)[^l1], steps like physically counting and reconciling items remain labor-intensive.

![Cycle-Count-Steps](./images/cycle-of-cycle-counting.png)
*Figure 1: The Cycle of Cycle Counting

This project investigates the potential of machine learning to automate or assist the inventory counting process. Utilizing the Amazon Bin Image Dataset, I explored pre-trained Convolutional Neural Network (CNN) models to count objects within bins. VGG19 emerged as the top-performing model with an overall accuracy of 53% and strong performance on per-class recall. This project demonstrates the feasibility of CNN-based object counting, paving the way for potential counting automation even in environments with less-than-ideal camera setups.


## Data Source

The Amazon Bin Image Dataset[^l3], comprising over 500,000 images and metadata from bins within an Amazon fulfillment center, served as our dataset. Each image is accompanied by a JSON file containing metadata describing the bin's contents, including stock-keeping units (SKUs) and their quantities.

Figure 2 shows a sample of the JSON message for `00001.jpg` image. Attribute `BIN_FCSKU_DATA` carries the SKUs (Stock Keeping Unit, i.e. product model) available in the bin. The SKU details, such as description, quantity, dimension, etc are provided in the Level -2 to -4 attributes. Attribute `EXPECTED_QUANTITY` shows how many objects available in the bin. This will be used as the label for the bin.

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

## Challenges and Assumptions

- **Computational Resources:** The dataset's size (3.34 GB) posed a computational challenge.
- **SKU Diversity:** The vast number of unique SKUs (>460,000) precluded the use of object detection approaches that rely on bounding boxes.
- **Image Quality:** Many images exhibited quality issues like blurriness, occlusions, and bundled items. While these impact model performance, they reflect real-world conditions and were included in the analysis.

![Low Quality Images](./images/AmazonBinImages-Challenges.png)


## Approach
To address these challenges, I focused on counting bins containing up to 5 items, aligning with the moderate difficulty level of the Amazon Bin Image Dataset (ABID) challenge[^l4]. I also ***assumed*** the metadata accurately reflected bin contents.



## Exploratory Data Analysis (EDA)

Key findings from the EDA include:
- The dataset comprises 536,434 images, each representing a single bin location within the fulfillment center.

![Amazon Bin Images](./images/AmazonBinImages.png)

Figure 3: Snapshot of the Bin Images

-  The dataset exhibits high SKU diversity, with 460,515 unique SKUs. Notably, 71.5% of SKUs are located in only one or two bin locations, and nearly 90% are found in five or fewer locations.

- Approximately 67.5% of the images depict bins containing five or fewer items, including empty bins.

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



## Data Preprocessing

I extracted item and quantity information from metadata to facilitate image filtering. Images were resized to 224x224 pixels and converted to NumPy arrays. Labels for model training were derived from the metadata's "expected quantity" field.  

To mitigate the impact of low-quality images on model training, a Laplacian variance filter was applied to the dataset. This filter identifies blurred images by measuring the variance of their Laplacian gradients. A variance threshold of 100 was selected, resulting in the removal of 15% of the images deemed to be excessively blurred.

Image Augmentation were considered during the model evaluation using the Keras' `ImageDataGenerator`. Given the marginal improvement gained from applying image augmentation techniques (e.g. horizontal/vertical flip, zoom, shear range, etc), in the expenses of 2-3 times model training time requires, Image Augmentation was removed fro the final model training. The huge amount of image data set in the Amazon Bin Image Challenge had sufficient model training data. 

## Machine Learning Model

Several pre-trained CNN models were explored (e.g., ResNet50, VGG19), utilizing transfer learning for efficiency. 
Ultimately, VGG19, pre-trained on ImageNet, was chosen for the final model. VGG19 stood out in the evaluation, for its consistent high overall accuracy, as well as relatively strong performance on per-class recall comparing to other models.  

| Model|Accuracy|
| --- | ---|
| MobilenetV2 |0.4145|
| EfficientNetV2M |0.1667|
| DenseNet|0.4578|
| VGG19|0.4513|
| InceptionResNetV2|0.4242|
| ResNet152V2|0.4212|



## Results and Future Work 
VGG19 achieved an overall accuracy of 55.3% and showed promising recall per class. While image quality limitations likely affected performance, the results demonstrate the potential of CNNs for automated inventory counting.

The model trained on the Amazon Bin Image Dataset could serve as a valuable starting point for object counting in other warehouse settings. By fine-tuning this pre-trained model with bin images specific to each warehouse, a rapid and cost-effective customization process can be achieved. This approach leverages the knowledge gained from the extensive Amazon dataset while tailoring the model to the unique characteristics of individual warehouses.



## Conclusion

This project successfully demonstrated the feasibility of utilizing machine learning, particularly CNNs, for automated inventory counting in fulfillment centers. The insights gained pave the way for further research and development in this area, potentially leading to significant efficiency gains in warehouse operations.


*Reference:*

[^l1]: https://www.bigcommerce.com/glossary/inventory-cycle-count/
[^l2]: https://www.netsuite.com.sg/portal/sg/resource/articles/inventory-management/using-inventory-control-software-for-cycle-counting.shtml
[^l3]: https://registry.opendata.aws/amazon-bin-imagery
[^l4]: https://github.com/silverbottlep/abid_challenge
