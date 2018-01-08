# Dice-coefficient-Loss-Caffe-Layer
Computes the dice coefficient loss for real-value regression task.<br>
*`(2xy+s)/(sum(x.^2)+sum(y.^2)+s)`*<br>
`x,y` are both vectors(***float32***). `s` means smooth term(default 1).<br>

**Please notice:**<br>
When `x,y` is not 0 or 1, `(2xy+s)/(sum(x.^2)+sum(y.^2)+s)` is not equal to `(2xy+s)/(sum(x)+sum(y)+s)`.

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
The same as `EuclideanLoss` layer. Only two bottom blobs setting is supported.
