'''
Baseline solution for the ACM Recsys Challenge 2017
using XGBoost

by Daniel Kohlsdorf
'''

import xgboost as xgb
import numpy as np
import multiprocessing

from model import *
from parser import *
import random
import numpy

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print(" --- Recsys Challenge 2017 Baseline --- ")

N_WORKERS         = 1
USERS_FILE        = "sample_users.csv"
ITEMS_FILE        = "sample_items.csv"
INTERACTIONS_FILE = "sample_interactions.csv"
TARGET_USERS      = "sample_target_users.csv"
TARGET_ITEMS      = "sample_target_items.csv"


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
(header_users, users) = select(USERS_FILE, lambda x: True, build_user, lambda x: int(x[0]))
(header_items, items) = select(ITEMS_FILE, lambda x: True, build_item, lambda x: int(x[0]))

builder = InteractionBuilder(users, items)
(header_interactions, interactions) = select(
    INTERACTIONS_FILE,
    lambda x: True,
    builder.build_interaction,
    lambda x: (int(x[0]), int(x[1]))
)


'''
4) Create target sets for items and users
'''
target_users = []
for line in open(TARGET_USERS):
    target_users += [int(line.strip())]
target_users = set(target_users)

target_items = []
for line in open(TARGET_ITEMS):
    target_items += [int(line.strip())]


targetIdx = {}
target_data = []
targetInteractions = {}
target_users_from_target_item = {}
idx = 0
for item in target_items:
    for user in target_users:
        x = Interaction(users[user], items[item], -1, -1, -1, -1, -1, -1)
        if x.title_match() > 0:
            target_data.append(x.features())
            targetIdx[idx] = (item,user)
            targetInteractions[(item,user)] = x
            if item in target_users_from_target_item.keys():
                target_users_from_target_item[item].append(user)
            else:
                target_users_from_target_item[item] = [user]
            idx+=1

target_data = np.array(target_data)
print(target_data.shape)
'''
2) Build recsys training data
'''
data    = np.array([interactions[key].features() for key in interactions.keys()])
print(data.shape)
mergedData = numpy.concatenate((data, target_data), axis=0)
print(mergedData.shape)

encoded_x = None
print(mergedData[:,:])
for i in range(0, mergedData.shape[1]):
    label_encoder = LabelEncoder()
    feature = label_encoder.fit_transform(mergedData[:,i])
    feature = feature.reshape(mergedData.shape[0], 1)
    # print(feature.shape)
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    # print(feature.shape)
    # print(feature)
    if encoded_x is None:
        encoded_x = feature
    else:
        print(i,encoded_x.shape)
        encoded_x = numpy.concatenate((encoded_x, feature), axis=1)
        print(i,encoded_x.shape)
print(data.shape)
encoded_data = encoded_x[:data.shape[0],:]
print(encoded_data.shape)

for i in range(target_data.shape[0]):
    item,user = targetIdx[i]
    targetInteractions[(item,user)].addEncodeData(encoded_x[data.shape[0]+i:data.shape[0]+i+1,:])
    # print(encoded_x[data.shape[0]+i,:].shape)
    # print(encoded_x.shape)

# i = 0
# for key in interactions.keys():
#     interactions[key].addEncodeData(encoded_x[i,:])
#     i+=1

print("data shape: : ", encoded_x.shape)

labels  = np.array([interactions[key].label() for key in interactions.keys()])
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
label_encoded_y = label_encoder.transform(labels)

# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y, test_size=test_size, random_state=seed)
#
# model = XGBClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))


dataset = xgb.DMatrix(encoded_data, label=label_encoded_y)
dataset.save_binary("recsys2017.buffer")


'''
3) Train XGBoost regression model with maximum tree depth of 2 and 25 trees
'''
evallist = [(dataset, 'train')]
param = {'bst:max_depth': 2, 'bst:eta': 0.1, 'silent': 1, 'objective': 'reg:linear' }
param['nthread']     = 4
param['eval_metric'] = 'rmse'
param['base_score']  = 0.0
num_round            = 25
bst = xgb.train(param, dataset, num_round, evallist)
bst.save_model('recsys2017.model')




'''
5) Schedule classification
'''
TH = 0

def classify_worker(item_ids, target_users, items, users, output_file, model):
    with open(output_file, 'w') as fp:
        pos = 0
        average_score = 0.0
        num_evaluated = 0.0
        for i in item_ids:
            encoded_x = None
            # data = []
            ids  = []
            if i not in target_users_from_target_item.keys():
                continue
            # build all (user, item) pair features based for this item
            for u in target_users_from_target_item[i]:
                # print(i,u)
                x = targetInteractions[(i,u)]
                # print(x.encoded.shape)
                if encoded_x is None:
                    encoded_x = x.encoded
                else:
                    encoded_x = numpy.concatenate((encoded_x, x.encoded), axis=0)
                ids  += [u]
                # print(encoded_x.shape)

                # x = Interaction(users[u], items[i], -1, -1, -1, -1, -1, -1)
                # if x.title_match() > 0:
                #     f = x.features()
                #     # data += [f]
                #     ids  += [u]
            if encoded_x is None:
                # predictions from XGBoost
                dtest = xgb.DMatrix(encoded_x)
                ypred = model.predict(dtest)
                # compute average score
                average_score += sum(ypred)
                num_evaluated += float(len(ypred))

                # use all items with a score above the given threshold and sort the result
                user_ids = sorted(
                    [
                        (ids_j, ypred_j) for ypred_j, ids_j in zip(ypred, ids) if ypred_j > TH
                    ],
                    key = lambda x: -x[1]
                )[0:99]
                # write the results to file
                if len(user_ids) > 0:
                    item_id = str(i) + "\t"
                    fp.write(item_id)
                    for j in range(0, len(user_ids)):
                        user_id = str(user_ids[j][0]) + ","
                        fp.write(user_id)
                    fp.write("\n")
                    fp.flush()

            # Every 100 iterations print some stats
            if pos % 100 == 0:
                try:
                    score = str(average_score / num_evaluated)
                except ZeroDivisionError:
                    score = str(0)
                percentageDown = str(pos / float(len(item_ids)))
                print(output_file + " " + percentageDown + " " + score)
            pos += 1

bucket_size = len(target_items) / N_WORKERS
start = 0
jobs = []
for i in range(0, N_WORKERS):
    stop = int(min(len(target_items), start + bucket_size))
    filename = "solution_" + str(i) + ".csv"
    process = multiprocessing.Process(target = classify_worker, args=(target_items[start:stop], target_users, items, users, filename, bst))
    jobs.append(process)
    start = stop

for j in jobs:
    j.start()

for j in jobs:
    j.join()

'''
6) Merge Solution Files
'''

s = open('solution.csv','w')
for i in range(N_WORKERS):
    f = open('solution_'+str(i)+'.csv','r')
    while True:
        line = f.readline()
        s.write(line)
        if not line: break
    f.close()
s.close()
