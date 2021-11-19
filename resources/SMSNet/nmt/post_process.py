# Copyright 2019 Korrawe Karunratanakul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import numpy as np

from nmt import input_config
from nmt.utils import file_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Reshape, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau


def cal_mass(line):
  if not line: return []
  # print(line)
  AAs = line.split(" ")
  aa_mass = []
  for aa in AAs:
    if aa in input_config.mass_AA and not (aa == '<s>' or aa == '</s>' or aa == '<unk>'):
      aa_mass.append(input_config.mass_AA[aa])
    else:
      return []

  return aa_mass


def compare_mass_without_mask(nmt_mass, ref_mass, label='nmt'):
    nmt_len = len(nmt_mass)
    ref_len = len(ref_mass)
    if abs(sum(nmt_mass) - sum(ref_mass)) > 0.05:
      if label == 'nmt':
        return [0] * nmt_len
      else:
        return [0] * ref_len
    
    pred_label = []
    
    i, j = 0, 0
    sum_nmt = 0.0
    sum_ref = 0.0
    while i < nmt_len and j < ref_len:
        if abs(sum_nmt - sum_ref) < 0.03 and abs(nmt_mass[i] - ref_mass[j]) < 0.0001:
            pred_label.append(1)
            sum_nmt += nmt_mass[i]
            sum_ref += ref_mass[j]
            i += 1
            j += 1
        else:
            if sum_nmt < sum_ref:
                sum_nmt += nmt_mass[i]
                if label == 'nmt':
                    pred_label.append(0)
                i += 1
            else:
                sum_ref += ref_mass[j]
                if label == 'ref':
                    pred_label.append(0)
                j += 1
    
    if label == 'nmt':
        while i < nmt_len:
            pred_label.append(0)
            i += 1
    
    if label == 'ref':
        while j < ref_len:
            pred_label.append(0)
            j += 1

    return pred_label
  
  
def prepare_label(file_data):
  labels = []
  for idx in range(file_data["length"]):
    ref_mass = cal_mass(file_data["ref_seqs"][idx])
    nmt_mass = cal_mass(file_data["nmt_seqs"][idx])
    if not ref_mass:
      labels.append([])
      continue
    result = compare_mass_without_mask(nmt_mass, ref_mass, label='nmt')
    labels.append(result)
  return labels


def prepare_feature(file_data, log_dir, labels_list=None):
  features = []
  labels = []
  positions = []
  for idx in range(file_data["length"]):
    probs = file_data["probs"][idx]
    if (labels_list and len(labels_list[idx]) <= 0 ) or probs[0] == '':
      continue

    probs = [np.exp(float(prob)) for prob in probs]
    
    # seq features
    seq_feat = []
    # num of aa in seq
    seq_feat.append(len(probs))
    # num of probs higher than 0.8 and 0.9
    seq_feat.append(sum([1 if prob > 0.7 else 0 for prob in probs]))
    seq_feat.append(sum([1 if prob > 0.8 else 0 for prob in probs]))
    seq_feat.append(sum([1 if prob > 0.9 else 0 for prob in probs]))
    # geometric mean of seq
    seq_feat.append(np.exp(np.sum(np.log(probs))/len(probs)))
    
    for posi in range(len(probs)):
      feat = seq_feat.copy()
      feat.append(posi/len(probs))
      # aa features
      aa_feat = [probs[posi+i] if (posi+i >= 0 and posi+i < len(probs)) else 1.0 for i in range(-2,2)]
      feat.extend(aa_feat)
      
      features.append(feat)
      if labels_list:
        labels.append(labels_list[idx][posi])
      else:
        positions.append((idx, posi))

  features = np.array(features)
  labels = np.array(labels)
  
  if labels_list:
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=42)
    feature_mean = np.mean(X_train, axis=0)
    feature_std = np.std(X_train, axis=0)
    np.savetxt(os.path.join(log_dir, 'feature_mean.txt'), feature_mean, fmt='%f')
    np.savetxt(os.path.join(log_dir,'feature_std.txt'), feature_std, fmt='%f')
    
    X_train = (X_train - feature_mean) / feature_std
    X_val = (X_val - feature_mean) / feature_std
    
    data_dict = {"train_x": X_train, "train_y": y_train,
                 "val_x": X_val, "val_y": y_val}
    
  else:
    feature_mean = np.loadtxt(os.path.join(log_dir, 'feature_mean.txt'), dtype=float)
    feature_std = np.loadtxt(os.path.join(log_dir, 'feature_std.txt'), dtype=float)
    features = (features - feature_mean) / feature_std
    data_dict = {"features": features, "positions": positions}

  return data_dict


def prepare_data_infer(file_data, log_dir):
  data_dict = prepare_feature(file_data, log_dir)
  return data_dict["features"], data_dict["positions"]


def prepare_data(file_data, log_dir):
  labels_list = prepare_label(file_data)
  data_dict = prepare_feature(file_data, log_dir, labels_list)

  return data_dict["train_x"], data_dict["val_x"], data_dict["train_y"], data_dict["val_y"]


def get_model(input_shape):
  input1 = Input(shape=(input_shape,))
  x = Dense(64, activation='relu')(input1)
  x = Dense(64, activation='relu')(x)
  out = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=input1, outputs=out)
  model.compile(optimizer=Adam(),
                loss='binary_crossentropy',
                metrics=['acc'])
  return model


def print_predicted_prob(test_probs, test_position_list, output_filename):
  with open(output_filename, 'w') as output_file:
    current_row = 0
    test_probs = np.squeeze(test_probs)
    out_str = ""
    for i, prob in enumerate(test_probs):
      current_out_row = test_position_list[i][0]
      while(current_row < current_out_row):
        current_row += 1
        output_file.write(out_str.strip() + "\n")
        out_str = ""
      
      out_str += str(np.log(prob)) + " "
    output_file.write(out_str.strip() + "\n")


def train(output_filename, prob_filename, tgt_filename, spectrum_filename, log_dir="log_post_process"):
  if not log_dir:
    log_dir="post_process"
  data_content = file_utils.read_output_file(output_filename, prob_filename, 
                                             tgt_filename, spectrum_filename)
  X_train, X_val, y_train, y_val = prepare_data(data_content, log_dir=log_dir)
  input_shape = X_train.shape[1]
  model = get_model(input_shape)
  model.summary()
  
  callbacks_list = [
    ModelCheckpoint(
        log_dir + "/post_processing_model_weight.h5",
        monitor = "val_loss",
        mode = 'min',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
    ),
  ]
  model.fit(X_train, y_train, epochs=30, batch_size=256, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks_list)
  


def rescore(output_filename, prob_filename, spectrum_filename,
            log_dir="post_process", output_path=None):
  if not log_dir:
    log_dir="post_process"
  if not output_path:
    output_path = os.path.join(log_dir, "rescore_prob")
  test_data_content = file_utils.read_output_file(output_filename, prob_filename,
                                                  None, spectrum_filename)
  test_x, test_positions = prepare_data_infer(test_data_content, log_dir=log_dir)
  
  input_shape = test_x.shape[1]
  model = get_model(input_shape)
  model.load_weights(log_dir + "/post_processing_model_weight.h5")
  test_probs = model.predict(test_x,verbose=1)
  
  print_predicted_prob(test_probs, test_positions, output_path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  
  parser.add_argument("--train", type="bool", nargs="?", const=True,
                      default=False,
                      help="Train new model.")
  parser.add_argument("--rescore", type="bool", nargs="?", const=True,
                      default=False,
                      help="Rescore with previously trained model.")
  parser.add_argument("--output_file", type=str, default=None,
                      help="Predition file from main model.")
  parser.add_argument("--prob_file", type=str, default=None,
                      help="Prob file from main model.")
  parser.add_argument("--tgt_file", type=str, default=None,
                      help="Target file for training.")
  parser.add_argument("--spectrum_file", type=str, default=None,
                      help="Source spectrum.")
  parser.add_argument("--logdir", type=str, default=None,
                      help="Directory to save or load model.")
  parser.add_argument("--output", type=str, default=None,
                      help="Output file path.")
  
  args = parser.parse_args()
  print(args)
  if args.train:
    print("  Training...")
    train(args.output_file, args.prob_file, args.tgt_file, args.spectrum_file, args.logdir)
    
  elif args.rescore:
    print("  Rescoring..")
    rescore(args.output_file, args.prob_file, args.spectrum_file, args.logdir, args.output)
    print("  Done")
