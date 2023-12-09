import numpy as np
import random
from torch.utils.data import Dataset


def process_vive(path, window_size=50, stride=1):
    genuine_data = np.load(path)
    users, sessions, timestamps, features = genuine_data.shape
    all_sessions = [s for s in range(sessions)]

    all_genuine = []
    all_imposter = []
    # for each user
    for user in range(users):
        other_users = [i for i in range(users) if i != user] # select all the other users
        user_genuine = []
        user_imposter = []
        # for each session
        for s in range(sessions):
            # apply sliding window
            start_pos = 0
            end_pos = window_size
            sub_genuine_sessions = []
            sub_imposter_sessions = []
            # for all valid windows
            while(end_pos < timestamps):
                sub_genuine_sessions.append(genuine_data[user, s, start_pos:end_pos, :])

                # generate imposter data
                random_user = random.choice(other_users)
                random_session = random.choice(all_sessions)
                if end_pos > genuine_data[random_user, random_session].shape[0]:
                    random_start_pos = random.choice([i for i in range(genuine_data[random_user, random_session].shape[0]-51)])
                    sub_imposter_sessions.append(genuine_data[random_user, random_session, random_start_pos:random_start_pos+50, :])
                else:
                    sub_imposter_sessions.append(genuine_data[random_user, random_session, start_pos:end_pos, :])

                # update start and end position of sliding window
                start_pos += stride
                end_pos += stride

            sub_genuine_sessions = np.stack(sub_genuine_sessions)
            sub_imposter_sessions = np.stack(sub_imposter_sessions)

            user_genuine.append(sub_genuine_sessions)
            user_imposter.append(sub_imposter_sessions)

        user_genuine = np.stack(user_genuine)
        user_imposter = np.stack(user_imposter)


        all_genuine.append(user_genuine)
        all_imposter.append(user_imposter)

    all_genuine = np.stack(all_genuine)
    all_imposter = np.stack(all_imposter)

    # [users, total sessions after augmented, timestamps, features]
    all_genuine = np.reshape(all_genuine, [users, -1, window_size, features])
    all_imposter = np.reshape(all_imposter, [users, -1, window_size, features])

    return all_genuine, all_imposter


class VIVE(Dataset):
    def __init__(self, path, user_id, window_size=50, stride=1):
        all_augmented_genuine_data, all_augmented_imposter_data = process_vive(path, window_size, stride)

        user_augmented_genuine_data = all_augmented_genuine_data[user_id]
        user_augmented_imposter_data = all_augmented_imposter_data[user_id]

        self.all_data = []
        for g in range(user_augmented_genuine_data.shape[0]):
            self.all_data.append((user_augmented_genuine_data[g], 1))
        for i in range(user_augmented_imposter_data.shape[0]):
            self.all_data.append((user_augmented_imposter_data[i], 0))

    def __getitem__(self, idx):
        data, label = self.all_data[idx]
        need_id = [0, 1, 2, 6]  # right hand xyz & trigger pressure
        data = data[:, need_id]
        return data, label 

    def __len__(self):
        return len(self.all_data)