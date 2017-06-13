#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const Dtype smooth = Dtype(0);
  Dtype sum_x;
  Dtype sum_y;
  caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(), &sum_x);
  caffe_gpu_dot(count, bottom[1]->gpu_data(), bottom[1]->gpu_data(), &sum_y);
  sum_ = sum_x + sum_y + smooth;
  caffe_gpu_dot(
            count, 
            bottom[0]->gpu_data(), 
            bottom[1]->gpu_data(), 
            &dice_);
  dice_ = Dtype(2) * dice_ + smooth;
  dice_ /= sum_;
  Dtype loss = Dtype(1) - dice_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = Dtype(1.0);
      const int index = (i == 0) ? 1 : 0;
      // LOG(INFO) << top[0]->cpu_diff()[0];
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / sum_;
      
      caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff());
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha*Dtype(-2),                 // alpha
          bottom[index]->gpu_data(),       // a
          alpha*dice_*Dtype(2),            // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DiceCoefLossLayer);

}  // namespace caffe
