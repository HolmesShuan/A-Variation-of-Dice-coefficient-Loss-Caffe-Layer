# Dice-coefficient-Loss-Caffe-Layer
Computes the dice coefficient loss for real-value regression task.<br>
There are two kinds of formulas:<br>
1. *`(2xy+s)/(sum(x)+sum(y)+s)`*
2. *`(2xy+s)/(sum(x.^2)+sum(y.^2)+s)`*

`x,y` are both vectors. `s` means smooth parameter.<br>
`.cpp` implemented the formula 1. `.cu` implemented the other. 

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
