# semantic-communication-
The code for the contributed paper：

Zhou Q, Li R, Zhao Z, et al. Adaptive Bit Rate Control in Semantic Communication with Incremental Knowledge-based HARQ[J]. arXiv preprint arXiv:2203.06634, 2022, accepted IEEE Open Journal of the Communications Society.

Zhou Q, Li R, Zhao Z, et al. Semantic communication with adaptive universal transformer[J]. IEEE Wireless Communications Letters, 2021, 11(3): 453-457,
accepted IEEE Wireless Communications Letters.

# Dataset:
For dataset, it is available online at http://www.statmt.org/europarl/

Choose the file "parallel corpus French-English, 194 MB, 04/1996-11/2011".

# Run the code:
To pre-process the data: preprocess_captions.py

To get the baseline: modeltrainbase.py

To get the baseline after quantification: modeltrainbasequantification1.py and modeltrainbasequantification2.py

Train with UT: modeltrainUT.py 

Train with IK-HARQ: modeltrainIKHARQ.py

Train with denoiser: modeltraindenoiser1.py and modeltraindenoiser2.py

Train with bit rate control: modeltrainmultibitratepart(123).py, modeltrainpolicynetpart(12).py

# For more detials:

Please refer to our paper for more details.

https://arxiv.org/pdf/2108.09119

https://arxiv.org/pdf/2203.06634
