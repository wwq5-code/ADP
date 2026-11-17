#!/bin/bash




echo "Start On_FMNIST/main_baseline_salun_for_different_unl_samples_928_15.txt"

python On_FMNIST/main_baseline_salun_for_different_unl_samples.py > On_FMNIST/main_baseline_salun_for_different_unl_samples_928_15.txt 2>&1


echo "Start On_FMNIST/main_unsupervised_baseline_928_15.txt"

python On_FMNIST/main_unsupervised_baseline.py > On_FMNIST/main_unsupervised_baseline_928_15.txt 2>&1


echo "Start On_SVHN/main_baseline_for_different_unl_samples_928_15.txt"

python On_SVHN/main_baseline_for_different_unl_samples.py > On_SVHN/main_baseline_for_different_unl_samples_928_15.txt 2>&1


echo "Start On_SVHN/main_baseline_salun_for_different_unl_samples_928_15.txt"

python On_SVHN/main_baseline_salun_for_different_unl_samples.py > On_SVHN/main_baseline_salun_for_different_unl_samples_928_15.txt 2>&1


echo "Start On_SVHN/main_unsupervised_baseline_928_15.txt"

python On_SVHN/main_unsupervised_baseline.py > On_SVHN/main_unsupervised_baseline_928_15.txt 2>&1


echo "Start On_TinyImageNet/main_baseline_for_different_unl_samples_928_15.txt"

python On_TinyImageNet/main_baseline_for_different_unl_samples.py > On_TinyImageNet/main_baseline_for_different_unl_samples_928_15.txt 2>&1


echo "Start On_TinyImageNet/main_baseline_salun_for_different_unl_samples_928_15.txt"

python On_TinyImageNet/main_baseline_salun_for_different_unl_samples.py > On_TinyImageNet/main_baseline_salun_for_different_unl_samples_928_15.txt 2>&1

echo "Start On_TinyImageNet/main_unsupervised_baseline_928_15.txt"

python On_TinyImageNet/main_unsupervised_baseline.py > On_TinyImageNet/main_unsupervised_baseline_928_15.txt 2>&1




echo "All finished."
