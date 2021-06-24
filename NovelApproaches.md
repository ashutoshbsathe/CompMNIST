# Novel approaches to try out

* Scene graph generation and understanding
  * [Jianwei Yang et al.](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jianwei_Yang_Graph_R-CNN_for_ECCV_2018_paper.pdf) -- generates a scene graph using Graph R-CNN. Assuming that our relationship is "next-to", this approach won't work at all because the graph will always become a circle making it indistinguishable from other graphs
  * Might be worth checking the author's [thesis](https://smartech.gatech.edu/handle/1853/62744) on the same
  * As a stark difference from **"scene"** understanding, our work focusses more on **"object"** understanding. In case of MNIST, it specifically exploits the border of digits to make better models
