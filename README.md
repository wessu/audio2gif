# audio2gif
A audio-to-gif GAN

Thie repository implement a audio-visual GAN model. 

Our code is based of StackGAN reference code
https://github.com/hanzhanggit/StackGAN-Pytorch/tree/master/code

What we implement/modified:
1. GAN:
     - Modify StackGAN to work with new data sources 
     - Implement WGAN options in StackGAN 
     - Implement two-stream network in StackGAN 
2. Audio:
     - All data preprocessing, feature extract
     - Implement and train AudioSet embedding
3. Other:
     - Evaluation code for incpetion score + Fine tue VGG net for inception score
4. Training:
     - Trained a lot of GANs ..... on both AudioSet and COCO (for testing)
    .   
