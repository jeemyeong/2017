import csv

USERS_FILE               = "users.csv"
ITEMS_FILE               = "items.csv"
INTERACTIONS_FILE        = "interactions_count.csv"
TARGET_USERS             = "targetUsers.csv"
TARGET_ITEMS             = "targetItems.csv"
SAMPLE_USER_FILE         = "sample_users.csv"
SAMPLE_ITEMS_FILE        = "sample_items.csv"
SAMPLE_INTERACTIONS_FILE = "sample_interactions.csv"
SAMPLE_TARGET_USERS      = "sample_target_users.csv"
SAMPLE_TARGET_ITEMS      = "sample_target_items.csv"
maxIdx = 10000
# interaction count
interaction_dict = {}
with open("interactions.csv", 'r') as csvFile:
    first = True
    reader = csv.reader(csvFile, delimiter='\t')
    i = 0
    for row in reader:
        i += 1
        if first:
            first = False
            header_interactions = ['recsyschallenge_v2017_interactions_final_anonym_training_export.item_id', 'recsyschallenge_v2017_interactions_final_anonym_training_export.user_id', 'recsyschallenge_v2017_interactions_final_anonym_training_export.0', 'recsyschallenge_v2017_interactions_final_anonym_training_export.1', 'recsyschallenge_v2017_interactions_final_anonym_training_export.2', 'recsyschallenge_v2017_interactions_final_anonym_training_export.3', 'recsyschallenge_v2017_interactions_final_anonym_training_export.4', 'recsyschallenge_v2017_interactions_final_anonym_training_export.5']
            continue

        user_id, item_id, interaction = (row[0], row[1], row[2])

        if item_id not in interaction_dict:
            interaction_dict[item_id] = {}

        if user_id not in interaction_dict[item_id]:
            interaction_dict[item_id][user_id] = {}
            interaction_dict[item_id][user_id]['0'] = 0
            interaction_dict[item_id][user_id]['1'] = 0
            interaction_dict[item_id][user_id]['2'] = 0
            interaction_dict[item_id][user_id]['3'] = 0
            interaction_dict[item_id][user_id]['4'] = 0
            interaction_dict[item_id][user_id]['5'] = 0

        interaction_dict[item_id][user_id][interaction] += 1

        if i % 500000 == 0:
            print(str(i)+' lines have been read.');

with open('interactions_count.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, delimiter='\t')
    writer.writerow(header_interactions)
    for item_id in interaction_dict:
        for user_id in interaction_dict[item_id]:
            writer.writerow([item_id, user_id, interaction_dict[item_id][user_id]['0'],
                            interaction_dict[item_id][user_id]['1'], interaction_dict[item_id][user_id]['2'],
                            interaction_dict[item_id][user_id]['3'], interaction_dict[item_id][user_id]['4'],
                            interaction_dict[item_id][user_id]['5']])

with open(USERS_FILE, 'r') as users_file:
    with open(SAMPLE_USER_FILE, 'w') as sample_users_file:
        print("Make Sample User File")
        first = True
        reader = csv.reader(users_file, delimiter='\t')
        i = 0
        for row in reader:
            i += 1
            if first:
                first = False
                sample_users_file.write("\t".join(row)+"\n")
                continue
            idx = int(row[0])
            if idx <= maxIdx:
                sample_users_file.write("\t".join(row)+"\n")
            if i % 500000 == 0:
                print(str(i)+' lines have been read.')

with open(ITEMS_FILE, 'r') as items_file:
    with open(SAMPLE_ITEMS_FILE, 'w') as sample_items_file:
        print("Make Sample Item File")
        first = True
        reader = csv.reader(items_file, delimiter='\t')
        i = 0
        for row in reader:
            i += 1
            if first:
                first = False
                sample_items_file.write("\t".join(row)+"\n")
                continue
            idx = int(row[0])
            if idx <= maxIdx:
                sample_items_file.write("\t".join(row)+"\n")
            if i % 500000 == 0:
                print(str(i)+' lines have been read.')


with open(INTERACTIONS_FILE, 'r') as interactions_file:
    with open(SAMPLE_INTERACTIONS_FILE, 'w') as sample_interactions_file:
        print("Make Sample Interaction File")
        first = True
        reader = csv.reader(interactions_file, delimiter='\t')
        i = 0
        for row in reader:
            i += 1
            if first:
                first = False
                sample_interactions_file.write("\t".join(row)+"\n")
                continue
            userIdx = int(row[0])
            itemIdx = int(row[1])
            if userIdx <= maxIdx and itemIdx <= maxIdx:
                sample_interactions_file.write("\t".join(row)+"\n")
            if i % 500000 == 0:
                print(str(i)+' lines have been read.')

with open(TARGET_USERS, 'r') as target_users:
    with open(SAMPLE_TARGET_USERS, 'w') as sample_target_users:
        print("Make Sample Target_Users File")
        first = True
        reader = csv.reader(target_users, delimiter='\t')
        i = 0
        for row in reader:
            i += 1
            userIdx = int(row[0])
            if userIdx <= maxIdx:
                sample_target_users.write("\t".join(row)+"\n")
            if i % 500000 == 0:
                print(str(i)+' lines have been read.')

with open(TARGET_ITEMS, 'r') as target_items:
    with open(SAMPLE_TARGET_ITEMS, 'w') as sample_target_items:
        print("Make Sample Target_Items File")
        first = True
        reader = csv.reader(target_items, delimiter='\t')
        i = 0
        for row in reader:
            i += 1
            userIdx = int(row[0])
            if userIdx <= maxIdx:
                sample_target_items.write("\t".join(row)+"\n")
            if i % 500000 == 0:
                print(str(i)+' lines have been read.')
