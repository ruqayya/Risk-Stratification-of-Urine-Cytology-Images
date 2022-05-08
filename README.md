# Risk-Stratification-of-Urine-Cytology-Images
<p align="center">
  <img src="https://github.com/ruqayya/Risk-Stratification-of-Urine-Cytology-Images/blob/main/etc/cell_overlay.png" width="350" title="Network prediction">
</p>

<p>This repository contains code for predicting different types of cell in urine cytology images. The list of classes are: Normal urothelial cells, Squamous cells, Inflammatory, Others, Atypical urothelial cells, Malignant urothelial cells and Debris. The illustration of our proposed method for cell detection and classification from a WSI is shown in figure below.
</p>

<p align="center">
  <img src="https://github.com/ruqayya/Risk-Stratification-of-Urine-Cytology-Images/blob/main/etc/system_flow.jpg" width="700" title="SystemFlowDiagram">
</p align="center">
<p style="font-size:11px">(a) ROI detection (b) patches of size 5000 x 5000 are extracted from ROI c) unit which process every patch and output the coordinates and predicted label of each candidate cell (c1) cell segmentation followed by connected component analysis (c2) patch extraction (c3) label prediction using a trained classiier.
</p>

<p>The current version of code works with jp2 whole slide images of urine cytology samples. A sample jp2 image will be downloaded if there are no images available for processing. A trained network checkpoint which was used to generate results for this study is also provided with the code. 
</p>

<p>
This work is published in Cytometry Part A, https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.24313.

Awan, R., Benes, K., Azam, A., Song, T.H., Shaban, M., Verrill, C., Tsang, Y.W., Snead, D., Minhas, F. and Rajpoot, N., 2021. Deep learning based digital cell profiles for risk stratification of urine cytology images. Cytometry Part A, 99(7), pp.732-742.
</p>
 






