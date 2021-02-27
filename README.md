# In_Silico_Labeling_CBIO_internship
Research Internship at CBIO (Mines ParisTech school) in Bio-imaging. 

Work on In Silico Labeling method. Goal : develop an image-to-image translation model able to generate a fluorescent microscopy image from its corresponding transmitted-light microscopy image. 
U-Net architecture + different training loss experiments (L1, Pix2pix from Phillip Isola et al. https://arxiv.org/pdf/1611.07004.pdf based on conditional GANs, content loss from Gatys et al. https://arxiv.org/pdf/1508.06576.pdf based on transfer learning for feature extraction) 

Second part : work on nuclei segmentation. We have investigated the use of In Silico Labeling for pre-training segmentation networks. We showed that using ISL model as a "pretext task" for cell nuclei segmentation could reduce the number of annotated data required without sacrificing performance. 

You can read more about our work in my master thesis (pdf masters_thesis). You will find the python coding part in src_code file. 
