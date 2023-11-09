# Instance-Agnostic and Practical Clean Label Backdoor Attack Method for Deep Learning Based Face Recognition Models
## Source Code of the paper "Instance-Agnostic and Practical Clean Label Backdoor Attack Method for Deep Learning Based Face Recognition Models". 
<p align="center" style="..."> 
  <img src="./img/proposed_method.png" alt="Proposed Method"/>
  <i><b>Figure 1.</b> Overall operation of the proposed attack method</i>
</p>

## Abstract
Backdoor attacks, which induce a trained model to behave as intended by an adversary for specific inputs, have recently emerged as a serious security threat in deep learning-based classification models. In particular, because a backdoor attack is executed solely by incorporating a small quantity of malicious data into a dataset, it poses   significant threat to authentication models, such as facial cognition systems. Depending on whether the label of the poisoned samples has been changed, backdoor attacks on deep learning-based face recognition methods are categorized into one of the two architectures: (1) corrupted label attack; and (2) clean label attack. Clean label attack methods have been actively studied because they can be performed without access to training datasets or training processes. However, the performance of previous clean label attack methods is limited in their application to deep learning-based face recognition methods because they only consider digital triggers with instance-specific characteristics. In this study, we propose a novel clean label backdoor attack, that solves the limitations of the scalability of previous clean label attack methods for deep learning-based face recognition models. To generate poisoned samples that are instance agnostic while including physical triggers, the proposed method applies three core techniques: (1) accessory injection, (2) optimization-based feature transfer, and (3) N:1 mapping for generalization. From the experimental results under various conditions, we demonstrate that the proposed attack method is effective for deep learning-based face recognition models in terms of the attack success rate on unseen samples.We also show that the proposed method not only outperforms the recent clean label attack methods, but also maintains a comparable level of classification accuracy when applied to benign data.

<p align="center" style="..."> 
  <img src="./img/comparison.png" alt="Comparison"/>
  <i><b>Figure 2.</b> Difference between previous clean label attack methods and the proposed attack method</i>
</p>

## Repository Artifacts
- `attack`: code folder related to clean label attack methods
- `model`: code folder related to target face recognition model
- `dataset`: dataset for generating poisoned samples, training the target model, and testing the each attack method 
- `1_convex_attack.py`: run file for performing convex polytope attack
- `2_bullseye_attack.py`: run file for performing bulleseye convex polytope attack
- `3_digital_attack.py`: run file for performing hidden trigger attack 
- `4_proposed_attack.py`: run file for performing proposed attack method
- `README.md`: light documentation of this repository

## Implementation Environment
- Linux: Ubuntu 16.04.7
- Python: 3.7.9
- Pytorch: 1.10.1+cu102
- matplotlib==3.5.2
- torchvision==0.11.2+cu102
- opencv-python==4.6.0.66
- imageio==2.19.3
- scipy==1.7.3

### Run Convex Polytope Attack 
```
python3 1_convex_attack.py
```
### Run Bullseye Convex Polytope Attack
```
python3 2_bullseye_attack.py
```
### Run Hidden Trigger Attack 
```
python3 3_digital_attack.py
```
### Run Proposed Attack Method
```
python3 4_proposed_attack.py
```


