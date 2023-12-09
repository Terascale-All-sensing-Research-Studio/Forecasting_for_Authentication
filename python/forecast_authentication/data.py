import os
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader


def process_vive2(np_array, window_size=50, stride=1):
    genuine_data = np_array
    users, sessions, timestamps, features = genuine_data.shape
    all_sessions = [s for s in range(sessions)]

    all_genuine = []
    all_imposter = []
    for user in range(users):
        other_users = [i for i in range(users) if i != user] # select all the other users
        user_genuine = []
        user_imposter = []
        for s in range(sessions):
            # apply sliding window
            start_pos = 0
            end_pos = window_size
            sub_genuine_sessions = []
            sub_imposter_sessions = []
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


class Fore_Auth(Dataset):
    def __init__(self, 
        root_path, 
        flag='train', 
        user_id=0, 
        stride=1,
        forecasting_sizes=None,
    ):

        if flag == 'train':
            d = np.load(os.path.join(root_path, "Vive1.npy"))
        else:
            d = np.load(os.path.join(root_path, "Vive2.npy"))

        self.seq_len, self.label_len, self.pred_len = forecasting_sizes if forecasting_sizes is not None else (30, 10, 10)

        # generate time features for each timestamp, from one to end
        self.num_time_feat = d.shape[2]
        time_feature = [i+1 for i in range(self.num_time_feat)]
        time_feature = np.asarray(time_feature)

        # add time feature into raw data (to the first column)
        d_add_time_feat = np.zeros([d.shape[0], d.shape[1], d.shape[2], d.shape[3]+1])
        d_add_time_feat[:, :, :, 1:] = d
        d_add_time_feat[:, :, :, 0] = time_feature

        # apply sliding window
        ws = self.seq_len + self.pred_len
        all_genuine_d, all_imposter_d = process_vive2(d_add_time_feat, window_size=ws, stride=stride)

        # extract data for the spcific user from 2 days
        user_genuine_d = all_genuine_d[user_id, :, :, :]
        user_imposter_d = all_imposter_d[user_id, :, :, :]

        self.data_2_use = []
        for g in range(user_genuine_d.shape[0]):
            self.data_2_use.append((user_genuine_d[g], 1))
        for i in range(user_imposter_d.shape[0]):
            self.data_2_use.append((user_imposter_d[i], 0))


    def __getitem__(self, idx):
        data, label = self.data_2_use[idx]
        
        need_id = [0, 1, 2, 3, 7]  # right hand xyz & trigger pressure
        data = data[:, need_id]

        forecasting_encoder_in = data[:self.seq_len, 1:]
        forecasting_decoder_in = data[self.seq_len-self.label_len:, 1:]
        forecasting_encoder_time_feat = data[:self.seq_len, 0]/self.num_time_feat - 0.5
        forecasting_decoder_time_feat = data[self.seq_len-self.label_len:, 0]/self.num_time_feat - 0.5

        return (
            forecasting_encoder_in,
            np.expand_dims(forecasting_encoder_time_feat,1), 
            forecasting_decoder_in,
            np.expand_dims(forecasting_decoder_time_feat,1),
            data,
            label
        )

    def __len__(self):
        return len(self.data_2_use)