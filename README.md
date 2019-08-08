# A Variation of Dice Coefficient Loss Layer
If you are interested in `Unet` or `Segmentation`, please jump to the `Related works`.    
## Target:
Compute the variation of dice coefficient loss for **`real-value`** regression task, such as super resolution. Formally, 

![equation](http://latex.codecogs.com/gif.latex?\ell(x,y)=\frac{2x^Ty+\epsilon}{x^Tx+y^Ty+\epsilon}) 

where ![equation](http://latex.codecogs.com/gif.latex?x,y) are both vectors in ***float32***. ![equation](http://latex.codecogs.com/gif.latex?\epsilon) referes to smooth term (default 1).   
## How to use?
```
layer {
  name: "loss"
  type: "DiceCoefLoss"
  bottom: "Deconv"
  bottom: "label"
  top: "loss"
}
```
The usage is the same as `EuclideanLoss` layer, restricted to `bottom_size==2`.
## Related works:
**Unet Dice Loss** for segmentation:

![equation](http://latex.codecogs.com/gif.latex?\ell(A,B)=\frac{2|AB|}{|A|+|B|})

**Source Code**:   
1. [Caffe Implementation I](https://github.com/yihui-he/caffe-dice-loss-layer).
2. [Caffe Implementation II](https://github.com/im-rishabh/Caffe-Dice-Loss-Layer).
3. [TensorFlow Unet with Dice Loss](https://github.com/jakeret/tf_unet).
