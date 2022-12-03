### Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization

<b>Jingtang Liang*, <a href='http://vinthony.github.io'>Xiaodong Cun*</a>, <a href='http://www.cis.umac.mo/~cmpun/'>Chi-Man Pun</a>, <a href='https://juew.org/'>Jue Wang</a> </b>

[Paper](https://arxiv.org/abs/2109.05750) | [Code Repo](https://github.com/stefanLeong/S2CRNet) | Demos (this repo)


<p style='text-align:center'><img src='https://user-images.githubusercontent.com/4397546/133198989-e4e2cc21-92fd-4f9b-a487-cdb05be3175e.png'><p>

<i>Image harmonization aims to modify the color of the composited region with respect to the specific background. Previous works model this task as a pixel-wise image-to-image translation using UNet family structures. However, the model size and computational cost limit the performability of their models on edge devices and higher-resolution images. To this end, we propose a novel spatial-separated curve rendering network(S<sup>2</sup>CRNet) for efficient and high-resolution image harmonization for the first time. In S<sup>2</sup>CRNet, we firstly extract the spatial-separated embeddings from the thumbnails of the masked foreground and background individually. Then, we design a curve rendering module(CRM), which learns and combines the spatial-specific knowledge using linear layers to generate the parameters of the pixel-wise curve mapping in the foreground region. Finally, we directly render the original high-resolution images using the learned color curve. Besides, we also make two extensions of the proposed framework via the Cascaded-CRM and Semantic-CRM for cascaded refinement and semantic guidance, respectively. Experiments show that the proposed method reduces more than 90% parameters compared with previous methods but still achieves the state-of-the-art performance on both synthesized iHarmony4 and real-world DIH test set. Moreover, our method can work smoothly on higher resolution images in real-time which is more than 10× faster than the existing methods. </i> 


> Our method runs very fast and can harmonized the image/videos in CPU in near real-time! We have tested the performance on both Linux/Mac M1.

## Install
Our method only requires minimal dependences, please check the `requirements.txt`.
```
pip install -r requirements.txt
```

## Image Harmonization Demo

please run the `demo.ipynb` for more details.

## Video Harmonization Demo

We use [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) for video matting and the local video camera as the source video.

run:

```
python video_harmonization_demo.py
```

will create a side-by-side view to show the comparsion of the original video, copy-paste directly and the harmonized video.


##### Citation
```
@misc{liang2021spatialseparated,
      title={Spatial-Separated Curve Rendering Network for Efficient and High-Resolution Image Harmonization}, 
      author={Jingtang Liang and Xiaodong Cun and Chi-Man Pun and Jue Wang},
      year={2021},
      eprint={2109.05750},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


##### Related Work

* [TIP 2020] [S<sup>2</sup>AM: Improving the Harmony of the Composite Image by Spatial-Separated Attention Module.](https://github.com/vinthony/s2am)
* [AAAI 2021] [Split then Refine: Sequential Attention-guided ResUNets for Blind Single Image Visible Watermark Removal](https://github.com/vinthony/deep-blind-watermark-removal)
