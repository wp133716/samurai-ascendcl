## 本文件包含用于将 ONNX 模型转换为 om（Ascend） 模型的示例命令。

- ### image_encoder
``` bash
atc --model=image_encoder.onnx --output=image_encoder --framework=5 --input_format=ND --soc_version=Ascend310 --input_shape="image:1,512,512,3" --out_nodes="high_res_features0;high_res_features1;low_res_features;vision_pos_embeds;pix_feat_with_mem"
```

- ### memory_attention
``` bash
atc --model=memory_attention_simplified.onnx --output=memory_attention --framework=5 --input_format=ND --soc_version=Ascend310 --input_shape="current_vision_feats:1024,1,256;current_vision_pos_embeds:1024,1,256;maskmem_feats:1024,-1,1,64;memory_pos_embed:1024,-1,1,64;obj_ptrs:-1,1,256;obj_pos:-1" --dynamic_dims="1,1,1,1;2,2,2,2;3,3,3,3;4,4,4,4;5,5,5,5;6,6,6,6;7,7,7,7;7,7,8,8;7,7,9,9;7,7,10,10;7,7,11,11;7,7,12,12;7,7,13,13;7,7,14,14;7,7,15,15;7,7,16,16" --out_nodes="pix_feat_with_mem"
```

- ### memory_encoder
```
atc --model=memory_encoder.onnx --output=memory_encoder --framework=5 --input_format=NCHW --soc_version=Ascend310 --input_shape="pix_feat:1024,1,256;mask_for_mem:1,1,512,512;object_score_logits:1, 1;is_mask_from_pts:1" --out_nodes="maskmem_features;maskmem_pos_enc"
```

- ### mask_decoder
```
atc --model=../onnx_model/mask_decoder.onnx --output=mask_decoder  --framework=5 --input_format=NCHW --soc_version=Ascend310 --input_shape="point_coords:1,2,2;point_labels:1,2;pix_feat_with_mem:1,256,32,32;high_res_features_0:1,32,128,128;high_res_features_0:1,64,64,64" --out_nodes="low_res_multimasks;high_res_multimasks;ious;obj_ptr;object_score_logits"
```
