#!/bin/bash


echo "Start On_CIFAR10/VIB_unl_rfu_fingerprint_eva_718"

python On_CIFAR10/VIB_unl_rfu_fingerprint_eva.py > On_CIFAR10/VIB_unl_rfu_fingerprint_eva_718 2>&1




echo "Start On_CIFAR10/VIB_unl_retrain_fingerprint_eva_718"

python On_CIFAR10/VIB_unl_retrain_fingerprint_eva.py > On_CIFAR10/VIB_unl_retrain_fingerprint_eva_718 2>&1




echo "Start On_CIFAR100/VIB_unl_rfu_fingerprint_eva_718"

python On_CIFAR100/VIB_unl_rfu_fingerprint_eva.py > On_CIFAR100/VIB_unl_rfu_fingerprint_eva_718 2>&1




echo "Start On_CIFAR100/VIB_unl_retrain_fingerprint_eva_718"

python On_CIFAR100/VIB_unl_retrain_fingerprint_eva.py > On_CIFAR100/VIB_unl_retrain_fingerprint_eva_718 2>&1




echo "Start On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_0_1"

export CIFAR100_DP_TARGET_EPSILON=0.1
python On_CIFAR100/VIB_unl_retrain_dpSGD.py > On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_0_1 2>&1

echo "Start On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_1"

export CIFAR100_DP_TARGET_EPSILON=1
python On_CIFAR100/VIB_unl_retrain_dpSGD.py > On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_1 2>&1


echo "Start On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_5"

export CIFAR100_DP_TARGET_EPSILON=5
python On_CIFAR100/VIB_unl_retrain_dpSGD.py > On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_5 2>&1


echo "Start On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_10"

export CIFAR100_DP_TARGET_EPSILON=10
python On_CIFAR100/VIB_unl_retrain_dpSGD.py > On_CIFAR100/VIB_unl_retrain_dpSGD_718_epsilon_10 2>&1




echo "All finished."