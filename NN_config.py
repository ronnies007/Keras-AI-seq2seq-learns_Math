# some paths
cfg.checkpoint_path = "T:/csv_data/checkpoints/"
cfg.log_path = "R:/temp/"
cfg.log_file = "training_log.txt"

# Parameters for the model and dataset.
cfg.training_size = 55040   # number of question/answer pairs to be generated before every epoch
cfg.digits = int(3)         # how many max. digits allowed in single number
cfg.reverse = False         # this was valid for only using addition in the original code
cfg.hidden_size_1 = 2048    # num of lstm layer dimension
cfg.hidden_size_2 = 2048    # I just played around with these numbers. after adding more math to the questions 
                            # ... it needed more neurons.
cfg.batch_size = 128        # number of pairs trained before network weights being updated
cfg.max_epochs = 100        # max number of epochs to be trained

# additional counters 
cfg.epoch_count = 1
cfg.step_counter = 0