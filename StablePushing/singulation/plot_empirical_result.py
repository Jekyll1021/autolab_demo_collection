import json
import numpy as np

from os import listdir, remove

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

def get_first_reachable(dic):
	dir_reach = []
	dir_dist = []
	dir_reach_length = []
	dir_change_in_pos = []
	for i in range(3):
		if dic[str(i)+" max reach"] > dic[str(i)+" dist to pushing line"]:
			dir_reach.append(i)
			dir_dist.append(dic[str(i)+" dist to pushing line"])
			dir_reach_length.append(dic[str(i)+" max reach"])
			dir_change_in_pos.append(dic[str(i)+" change of pos"])
	return dir_reach, dir_dist, dir_reach_length, dir_change_in_pos

def get_second_reachable(dic):
	second_reach = []

	dir_reach, dir_dist, dir_reach_length, dir_change_in_pos = get_first_reachable(dic)
	for i in range(3):
		for j in dir_reach:
			if i != j:
				if dic[str(min(i, j))+" to "+str(max(i, j))+" max reach"] > dic[str(min(i, j))+" to "+str(max(i, j))+" dist"]:
					second_reach.append(i)
	return list(set(second_reach) - set(dir_reach))

def get_balance(dic):
	s = 0
	for i in range(3):
		s += dic[str(i)+" side of pushing line"]
	return s

def get_normalized_distribution_single_config(folder_path, thres):
	mean_sep = []
	mean_sep_before_thres = []

	for filename in listdir(folder_path):
		with open(folder_path+"/"+filename) as f:
			dic = json.load(f)
			curr_dir_reach, curr_dir_dist, curr_dir_reach_length, curr_dir_change_in_pos = get_first_reachable(dic)
			first_contact = dic["first contact object"]
			first_contact_pos_change = 0
			if first_contact != -1:
				first_contact_pos_change = dic[str(first_contact)+" change of pos"]
			
			if first_contact_pos_change > 0:
				mean_sep_before_thres.append(dic["mean separation after push"] / 3)

			if len(curr_dir_reach) > 0 and sum(curr_dir_change_in_pos) > 0 and first_contact_pos_change > 0 and first_contact_pos_change > thres:
				mean_sep.append(dic["mean separation after push"] / 3)

	mean = np.mean(mean_sep_before_thres)
	std = np.std(mean_sep_before_thres)
	if len(mean_sep) <= 0 or std == 0 or std == np.nan or std == np.inf:
		return None, None, None
	
	return (np.array(mean_sep) - mean) / std, mean, std

if __name__ == "__main__":
	for filename in listdir("range_log"):
		with open("range_log/"+filename, 'r') as f:
			dic = json.load(f)
			if dic["first contact object"] == -1:
				remove("range_log/"+filename)

	# mean_sep = []

	# for filename in listdir("log/single/"):
	# 	with open("log/single/"+filename) as f:
	# 		dic = json.load(f)
	# 		curr_dir_reach, curr_dir_dist, curr_dir_reach_length, curr_dir_change_in_pos = get_first_reachable(dic)
	# 		first_contact = dic["first contact object"]
	# 		first_contact_pos_change = 0
	# 		if first_contact != -1:
	# 			first_contact_pos_change = dic[str(first_contact)+" change of pos"]

	# 		if len(curr_dir_reach) > 0 and sum(curr_dir_change_in_pos) > 0 and first_contact_pos_change > 6:
	# 			mean_sep.append(dic["mean separation after push"] / 6)
	# plt.hist(mean_sep, bins=np.arange(4, 10, 0.1))
	# plt.title("distribution of push results: first contact position change thres = 6")
	# plt.savefig("single7.jpg")
	# plt.close()

	# all_mean = {}

	# for thr in range(0, 7):
	# 	all_mean[thr] = []


	# for folder_path in listdir("log/"):
	# 	if len(folder_path.split(".")) != 2:
	# 		print(folder_path)

	# 		for thr in range(0, 7):
	# 			norm_lst, mean, std = get_normalized_distribution_single_config("log/"+folder_path, thr)
	# 			if norm_lst is not None: 
	# 				# plt.hist(norm_lst, bins=np.arange(-3, 3, 0.1))
	# 				# plt.title("distribution of "+folder_path+" push results: first contact position change thres = "+str(thr))
	# 				# plt.savefig("log/"+folder_path+"_thr"+str(thr)+".jpg")
	# 				all_mean[thr].append(np.mean(norm_lst))
	# 				# plt.close()

	# for thr in range(0, 7):
	# 	plt.hist(all_mean[thr], bins=np.arange(-0.5, 3, 0.1), alpha=0.4, label="threshold = "+str(thr))

	# plt.title("distribution of all push results: first contact position change thres = "+str(thr))
	# plt.legend(loc='upper right')
	# plt.savefig("log/distribution_change.jpg")
	# plt.close()

	# for thr in range(0, 7):
	# 	weights = np.ones_like(all_mean[thr])/float(len(all_mean[thr]))
	# 	plt.hist(all_mean[thr], weights=weights, bins=np.arange(-0.5, 3, 0.1), alpha=0.4, label="threshold = "+str(thr))
	# plt.title("distribution by prob of all push results: first contact position change thres = "+str(thr))
	# plt.legend(loc='upper right')
	# plt.savefig("log/distribution_change_prob.jpg")
	# plt.close()

	# strong_push = []
	# for thr in range(4, 7):
	# 	strong_push.extend(all_mean[thr])

	# high_thres = np.percentile(strong_push, 90)
	# low_thres = np.percentile(strong_push, 10)

	# x = {"good push mean sep thres": high_thres, "bad push mean sep thres": low_thres}
	# with open('log/all_env_thres.json', 'w') as f:
	# 		json.dump(x, f)



	# for thr in range(0, 7):
	# 	norm_lst, mean, std = get_normalized_distribution_single_config("log/18_random", thr)
	# 	if norm_lst is not None: 
	# 		plt.hist(norm_lst, bins=np.arange(-3, 3, 0.1))
	# 		plt.title("distribution of env 18 push results: first contact position change thres = "+str(thr))
	# 		plt.savefig("log/18_thr"+str(thr)+".jpg")
	# 		plt.close()
		




	



	




