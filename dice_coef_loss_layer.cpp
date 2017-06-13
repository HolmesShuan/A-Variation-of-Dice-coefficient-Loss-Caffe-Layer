#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  multiplier_.ReshapeLike(*bottom[0]);
  add_.ReshapeLike(*bottom[0]);
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_add(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      add_.mutable_cpu_data());
  sum_ = caffe_cpu_dot(count, add_.cpu_data(), multiplier_.cpu_data()) + Dtype(1);
  dice_ = Dtype(2) * caffe_cpu_dot(count, bottom[0]->cpu_data(), bottom[1]->cpu_data()) + Dtype(1);
  dice_ /= sum_;
  Dtype loss = Dtype(1) - dice_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = Dtype(1.0);
      const int index = (i == 0) ? 1 : 0;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / sum_;
      // LOG(INFO) << top[0]->cpu_diff()[0];
      caffe_set(bottom[i]->count(), Dtype(1), bottom[i]->mutable_cpu_diff());
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha*Dtype(-2),                 // alpha
          bottom[index]->cpu_data(),       // a
          alpha*dice_,                     // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiceCoefLossLayer);
#endif

INSTANTIATE_CLASS(DiceCoefLossLayer);
REGISTER_LAYER_CLASS(DiceCoefLoss);

}  // namespace caffe
