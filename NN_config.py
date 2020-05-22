# some paths
checkpoint_path = "T:/csv_data/checkpoints/"
log_path = "R:/temp/"
log_file = "training_log.txt"

# Parameters for the model and dataset.
training_size = 55040   # number of question/answer pairs to be generated before every epoch
digits = int(3)         # how many max. digits allowed in single number
reverse = False         # this was valid for only using addition in the original code
hidden_size_1 = 2048    # num of lstm layer dimension
hidden_size_2 = 2048    # I just played around with these numbers. after adding more math to the questions 
                            # ... it needed more neurons.
batch_size = 256        # number of pairs trained before network weights being updated
max_epochs = 100        # max number of epochs to be trained

# additional counters 
epoch_count = 1
step_counter = 0
