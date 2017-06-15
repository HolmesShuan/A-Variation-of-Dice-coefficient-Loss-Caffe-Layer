# Dice-coefficient-Loss-Caffe-Layer
Computes the dice coefficient loss for real-value regression task.<br>
*`(2xy+s)/(sum(x.^2)+sum(y.^2)+s)`*<br>
`x,y` are both vectors.`s` means smooth term(default 1).<br>

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
Adding `DICE_COEF_LOSS` at caffe.proto line ~1200.
```
// DEPRECATED: use LayerParameter.
message V1LayerParameter {
  repeated string bottom = 2;
  repeated string top = 3;
  optional string name = 4;
  repeated NetStateRule include = 32;
  repeated NetStateRule exclude = 33;
  enum LayerType {
    NONE = 0;
    ABSVAL = 35;
    ACCURACY = 1;
    ARGMAX = 30;
    BNLL = 2;
    CONCAT = 3;
    CONTRASTIVE_LOSS = 37;
    CONVOLUTION = 4;
    DATA = 5;
    DECONVOLUTION = 39;
    DROPOUT = 6;
    DUMMY_DATA = 32;
    DICE_COEF_LOSS = 50; # HERE!
    EUCLIDEAN_LOSS = 7;
    ...
```
