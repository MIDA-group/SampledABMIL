""" Area under the receiver operating characteristic curve (AUC) at a bag and instance levels. The code can require changes if not the same format of folder names is used. ### indicates where the changes might be required.  """

import numpy as np
import os
import sklearn
from sklearn import metrics
import time
from time import perf_counter


def list_files_in_folder(image_folder):
    """Lists file names in a given directory"""
    list_of_files = []
    for file in os.listdir(image_folder):
        if os.path.isfile(os.path.join(image_folder, file)):
            list_of_files.append(file)
    return list_of_files

def create_save_dir(direct, name_subdirectory):
    if not os.path.exists(os.path.join(direct, name_subdirectory)):
        print('make dir')
        os.mkdir(os.path.join(direct, name_subdirectory))
    return os.path.join(direct, name_subdirectory)


def compute_metrics(subfolder_valid_approach, num_test_bags, weights_folder, root, num_samples):
    start_t = perf_counter() 
    count_TP=0;count_TN=0;count_FP=0;count_FN=0; 
    all_true_labels=[]; all_pred_labels=[]
    AUC_all_bags=[]; AUC_positive_bags=[]; 
    true_all_bags_label=[]
    for bag_num in range(0, num_test_bags):
        bag_folder = os.path.join(weights_folder, subfolder_valid_approach, str(bag_num).zfill(4)) ###
        list_weights_bag = list_files_in_folder(bag_folder)
        if not os.path.exists(os.path.join(root, 'test', 'negative', str(bag_num).zfill(4))): ###
            bag_folder_test = os.path.join(root, 'test', 'positive', str(bag_num).zfill(4)) ###
            true_bag_label = 1
        else:
            bag_folder_test = os.path.join(root, 'test', 'negative', str(bag_num).zfill(4)) ###
            true_bag_label = 0
            
        # Bag level label
        all_pred_sample_labels=[]
        for s in range(num_samples):
            for i in range(len(list_weights_bag)):
                if s<100 and (not list_weights_bag[i][-21:-20].isnumeric()):
                    if list_weights_bag[i][-20:-18]==str(s).zfill(2):
                        pred_sample_label = list_weights_bag[i][-8:-7] ###
                        all_pred_sample_labels.append(pred_sample_label)
                        break # all instances in sample have the same bag label
                elif s>=100:
                    if list_weights_bag[i][-21:-18]==str(s).zfill(2):
                        pred_sample_label = list_weights_bag[i][-8:-7] ###
                        all_pred_sample_labels.append(pred_sample_label)                    
                        break # all instances in sample have the same bag label
                    
        majority_bag_label_from_all_samples_for_one_bag = np.mean(np.asarray(all_pred_sample_labels).astype(np.int))
        pred_bag_label = 0
        if majority_bag_label_from_all_samples_for_one_bag>=0.5: #majority label bag level ###
            pred_bag_label=1  
            
        # Bag level metrics
        if true_bag_label==1 and pred_bag_label==1:
            count_TP+=1
        elif true_bag_label==1 and pred_bag_label==0:
            count_FN+=1
        elif true_bag_label==0 and pred_bag_label==1:
            count_FP+=1
        elif true_bag_label==0 and pred_bag_label==0:
            count_TN+=1     
            
        all_true_labels.append(true_bag_label); all_pred_labels.append(pred_bag_label)
            
        # Instance level label
        weights_all_samples_for_one_bag=[]; names_all_samples_for_one_bag=[]
        for s in range(num_samples):
            all_inst_weights_in_one_sample=[]; all_inst_names_in_one_sample=[]
            for i in range(len(list_weights_bag)):       
                if s<100 and (not list_weights_bag[i][-21:-20].isnumeric()):
                    if list_weights_bag[i][-20:-18]==str(s).zfill(2):
                        coeff = np.load(os.path.join(bag_folder, list_weights_bag[i]))
                        all_inst_weights_in_one_sample.append(coeff)
                        all_inst_names_in_one_sample.append(list_weights_bag[i])
                elif s>=100:
                    if list_weights_bag[i][-21:-18]==str(s).zfill(2):
                        coeff = np.load(os.path.join(bag_folder, list_weights_bag[i]))
                        all_inst_weights_in_one_sample.append(coeff)
                        all_inst_names_in_one_sample.append(list_weights_bag[i])

            min_coef = np.min(np.asarray(all_inst_weights_in_one_sample))
            max_coef = np.max(np.asarray(all_inst_weights_in_one_sample))
            norm_coeff = [(coef-min_coef)/(max_coef-min_coef+10e-12) for coef in all_inst_weights_in_one_sample] 
            weights_all_samples_for_one_bag.append(norm_coeff)
            names_all_samples_for_one_bag.append(all_inst_names_in_one_sample)

        all_test_img_names_in_bag = list_files_in_folder(bag_folder_test)
        average_weights_all_instances_one_bag=[]; majority_label_all_instances_one_bag=[]
        all_images_true_instance_label=[]
        for j in range(len(all_test_img_names_in_bag)):
            one_image_pred_sample_label=[];  one_image_weights=[]
            img_name = all_test_img_names_in_bag[j]  
            for s in range(len(names_all_samples_for_one_bag)):
                temp_names = names_all_samples_for_one_bag[s]
                temp_weights = weights_all_samples_for_one_bag[s]
                for t in range(len(temp_names)):
                    if img_name in temp_names[t]:
                        one_image_weights.append(temp_weights[t])
                        pred_sample_label = temp_names[t][-8:-7]  ###
                        true_instance_label = temp_names[t][-17:-16]   ###
                        one_image_pred_sample_label.append(pred_sample_label)
                        
            if not one_image_pred_sample_label:
                print('NO EVALUATION FOR IMAGE: ', j)
            all_images_true_instance_label.append(true_instance_label)
            arr_one_image_pred_sample_label = np.asarray(one_image_pred_sample_label).astype(np.int)
            if not one_image_pred_sample_label:
                print(arr_one_image_pred_sample_label)
            count1 = np.count_nonzero(arr_one_image_pred_sample_label == 1)
            if count1>np.around(len(arr_one_image_pred_sample_label)/2):
                majority_pred_samples_label_one_img = 1
            else:
                majority_pred_samples_label_one_img = 0
            count_weights=0; sum_weights=0
            for i in range(len(arr_one_image_pred_sample_label)):
                if arr_one_image_pred_sample_label[i]==majority_pred_samples_label_one_img:
                    sum_weights += one_image_weights[i]
                    count_weights +=1
            average_weight = sum_weights/(count_weights+1e-12)
            average_weights_all_instances_one_bag.append(average_weight)
            majority_label_all_instances_one_bag.append(majority_pred_samples_label_one_img)

        arr = np.asarray(average_weights_all_instances_one_bag)
        maj_arr = np.asarray(majority_label_all_instances_one_bag) #maj. bag level for each instance

        # Instance level metrics
        all_TPR=[]; all_FPR=[]
        threshold_range = np.arange(0,1+0.001,0.001)
        for th in range(len(threshold_range)):
            count_TP_ins=0;count_TN_ins=0;count_FP_ins=0;count_FN_ins=0
            Threshold = threshold_range[th]
            for k in range(len(arr)):
                if arr[k]>=Threshold and maj_arr[k]==1 and all_images_true_instance_label[k]==str(1):
                    count_TP_ins+=1
                elif arr[k]>=Threshold and maj_arr[k]==1 and all_images_true_instance_label[k]==str(0):
                    count_FP_ins+=1          
                elif (arr[k]>=Threshold and maj_arr[k]==0 and all_images_true_instance_label[k]==str(0)) or \
                (arr[k]<Threshold and all_images_true_instance_label[k]==str(0)):
                    count_TN_ins+=1      
                elif (arr[k]>=Threshold and maj_arr[k]==0 and all_images_true_instance_label[k]==str(1)) or \
                (arr[k]<Threshold and all_images_true_instance_label[k]==str(1)):
                    count_FN_ins+=1              

            # TPR, FPR for a certain threshold
            TPR = count_TP_ins/(count_TP_ins+count_FN_ins+1e-12)
            FPR = count_FP_ins/(count_FP_ins+count_TN_ins+1e-12)
            all_TPR.append(TPR); all_FPR.append(FPR)

        AUC_bag = sklearn.metrics.auc(np.asarray(all_FPR), np.asarray(all_TPR))
        AUC_all_bags.append(np.around(AUC_bag, decimals=5))
        if true_bag_label==1: #for positive bags
            AUC_positive_bags.append(np.around(AUC_bag, decimals=5))

    fpr, tpr, thres = sklearn.metrics.roc_curve(np.asarray(all_true_labels), np.asarray(all_pred_labels))
    AUC_bag_level = sklearn.metrics.auc(fpr, tpr)
    e = perf_counter() - start_t
    print("Elapsed time during the whole program in seconds:", e)

    print('AUC bag level', "%.3f" % AUC_bag_level)
    print('Bag level confusion matrix', [count_TP,count_FP,count_FN,count_TN])
    print('AUC for positive bags, instance level:', np.around(np.sum(np.asarray(AUC_positive_bags))/(num_test_bags/2), decimals=3))