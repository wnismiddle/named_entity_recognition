
class mlp_config(object):
    root_path = 'D://Program Files/Python/workspace/named_entity_recognition'
    hidden_units = [300, 300]
    train_epochs = 5
    learning_rate = 0.05
    tf_path = root_path + '/data/tmp/models/mlp.ckpt'
    batch_size = 128
    n_classes = 7
    n_input = 700
    labels = []
    tag_dict = {}



