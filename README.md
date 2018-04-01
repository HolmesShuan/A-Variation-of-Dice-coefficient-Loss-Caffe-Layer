# A Variation of Dice Coefficient Loss Layer
Interest in `Unet` or `Segmentation` may jump to `Related works`.    
## Target:
Compute the variation of dice coefficient loss for **`real-value`** regression task, such as super resolution. Mathematically,   
```matlab
Loss = (2x^Ty+e)/(x^Tx+y^Ty+e)
```
where `x,y` are both vectors in ***float32***. `e` referes to smooth term (default 1).   
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
```
Dice(A,B) = 2|AB|/(|A|+|B|)
```
**Source Code**:   
1. [Caffe Implementation I](https://github.com/yihui-he/caffe-dice-loss-layer).
2. [Caffe Implementation II](https://github.com/im-rishabh/Caffe-Dice-Loss-Layer).
3. [TensorFlow Unet with Dice Loss](https://github.com/jakeret/tf_unet).
