This is the official code of 'Universal Detection of Backdoor Attacks via Density-based Clustering and Centroids Analysis'.
Please download it from 'https://arxiv.org/abs/2301.04554'.

ABSTRACT:
In the paper, we propose a Universal Defence based on Clustering and Centroids Analysis (CCA-UD) against backdoor attacks.
The goal of the proposed defence is to reveal whether a Deep Neural Network model is subject to a backdoor attack by
inspecting the training dataset. CCA-UD first clusters the samples of the training set by means of density-based
clustering. Then, it applies a novel strategy to detect the presence of poisoned clusters. The proposed strategy is
based on a general misclassification behaviour obtained when the features of a representative example of the analysed
cluster are added to benign samples. The capability of inducing a misclassification error is a general characteristic of
poisoned samples, hence the proposed defence is attack-agnostic. This mask a significant difference with respect to
existing defences, that, either can defend against only some types of backdoor attacks, e.g., when the attacker corrupts
the label of the poisoned samples, or are effective only when some conditions on the poisoning ratios adopted by the
attacker or the kind of triggering pattern used by the attacker are satisfied. Experiments carried out on several
classification tasks, considering different types of backdoor attacks and triggering patterns, including both local and
global triggers, reveal that the proposed method is very effective to defend against backdoor attacks in all the cases,
always outperforming the state of the art techniques.

Structure of this project:
This project implements three dataset-level defence methods against the backdoor attacks: AC[1], CI[2], and CCA-UD. You
could find the relative codes from the corresponding folders.

How to use the code:
In each code of the method, we provide a demo to show how to use our code.

Refenence:
[1] Chen, et al. "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering." SafeAI@ AAAI. 2019.
[2] Xiang, et al. "A benchmark study of backdoor data poisoning defenses for deep neural network classifiers and a novel defense." 2019 MLSP. IEEE, 2019.