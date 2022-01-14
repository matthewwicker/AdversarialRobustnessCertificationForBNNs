from datasets import Dataset

datasets = ["boston1","energy1", "concrete1","powerplant1", "yacht1", "wine1", "kin8nm1", "naval1"]

for dataset_string in datasets:
    data = Dataset(dataset_string)

    X_train = data.train_set.train_data
    y_train = data.train_set.train_labels.reshape(-1,1)

    X_test = data.test_set.test_data
    y_test = data.test_set.test_labels.reshape(-1,1)
    print("Sucessfully loaded dataset %s \t \t with train shapes %s, %s"%(dataset_string, X_train.shape, y_train.shape))
