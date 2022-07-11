import os
import torch
import json
import yaml
import pickle

from models.random_clf import RandomClassifier
from models.random_attn_clf import RandomAttentionClassifier

from data_access.locations import LOC

# This modules holds utility for saving and loading of relevant data for the Experiment class
# TODO add other persisting (e.g. for dataset modules and visualization modules)

# generates hashcodes for the experiment class
def hash_id(exp, as_str=True):
    dirs = next(os.walk(LOC['experiments_dir']))[1]
    hash_id = abs(hash(exp))
    if hash_id in dirs: return False
    else: 
        if as_str: return str(hash_id)
        else: return hash_id

# save any model that is part of the experiment class
def save_model(exp):
    assert exp.model != None
    model_save_loc = LOC['models_dir'] + str(exp.eid) + '.pth'
    # TODO
    #   dirty fix: remove unwanted entries from the ordereddict: 
    #   also this only works for the linear layer...
    #   do it similarly to eval.efficiency_metrics() w. old & new and delete difference (mb init an empty model just for the state dict!)
    unwanted = ["lin.linear.input_shape", "lin.linear.output_shape", "lin.linear.parameter_quantity", "lin.linear.inference_memory", "lin.linear.MAdd", "lin.linear.duration", "lin.linear.Flops", "lin.linear.Memory"]
    state_dict = exp.model.state_dict()
    for u in unwanted: state_dict.pop(u)
    torch.save(state_dict, model_save_loc)
    return model_save_loc

# only for predictions of experiment class
def save_json(exp, obj, type=None): # type = 'preprocessed_data' | 'train_predictions' | 'predictions'
    if obj == None: return None
    if type == 'train_predictions':
        save_loc = LOC['predictions'] + str(exp.eid) + '_train.jsonl'
    elif type == 'test_predictions':
        save_loc = LOC['predictions'] + str(exp.eid) + '_test.jsonl'
    else:
        print('json type "' + type + '" unknown')
        assert False

    with open(save_loc, 'w') as outfile:
        json.dump(obj, outfile)
    return save_loc

# only for experiment yamls!
def save_yaml(exp, dic):
    assert exp.eid != None
    filename = LOC['experiments_dir'] + str(exp.eid) + '.yaml'

    with open(filename, 'w') as file:
        yaml_file = yaml.dump(dic, file)
    return yaml_file

# save pickles of the experiment class (mainly visualization data)
def save_pickle(exp, obj, save_loc=LOC['viz_data_dir']):
    filename = str(exp.eid) + '.pickle'

    with open(save_loc + filename, 'wb') as f:
        pickle.dump(obj, f)
    return save_loc + filename

def model_factory(type:str, parameters:{}=None, path:str=None):
    model = None

    if type == 'RandomClassifier':
        model = RandomClassifier(parameters['random_seed'])
    elif type == 'RandomAttentionClassifier':
        model = RandomAttentionClassifier(parameters['random_seed'])

    if path: 
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
    return model

# loads jsons for experiment class (mostyle preprocessed or prediction data)
def load_json(path:str):
    if path == None: return None
    data = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            data.append(result)
    return data

# loads pickles (mostly viz data)
def load_pickle(path:str):
    if path == None: return None
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

# parses tensor into list format (required by .yaml)
def parse_tensor_list(tensor_list:[]):
    return [x.squeeze().tolist() for x in tensor_list]