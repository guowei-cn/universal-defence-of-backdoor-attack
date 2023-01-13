Please download the feature hdf5 and model file from this link https://drive.google.com/file/d/1a8P5RzAIqeOC8XPHc2hDXSBePFlJo18Q/view?usp=sharing

After unzip, you could find three folders
1. The 'clean_label_gu_trigger' folder includes the backdoor model 'clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.pt' 
generated via: clean-label poisoning strategy with gu trigger, target class as 2 and poisoning ratio 0.3598428365005759.
The corresponding features are stored in 'feature_clean_tri_name_gu_10_types-1_target_class_2_poison_ratio_0.3598428365005759.hdf5'

2. The 'corrupted_label_gu_trigger' folder includes the backdoor model 'corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.pt'
generated via: corrupted-label posioning strategy with gu trigger, target class as 0 and poisoning ratio 0.0359842836500576.
THe corresponding features are stored in 'feature_corrupted_tri_name_gu_10_types-1_target_class_0_poison_ratio_0.0359842836500576.hdf5'

3 The 'corrupted_label_ramp_trigger' folder includes the backdoor model 'corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.pt'
generated via: corrupted-label poisoning strategy with ramp trigger, target class as 0 and poisoning ratio 0.0359842836500576
The corresponding features are stored in 'feature_corrupted_tri_name_ramp_target_class_0_poison_ratio_0.0359842836500576.hdf5'

more details please check our paper.
