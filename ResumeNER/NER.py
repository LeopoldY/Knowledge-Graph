from Optimizer import train, test

data_path = "/Users/leopold/Developer/Knowledge-Graph/ResumeNER/data/"
model_save_path = './model/model.crfsuite'

train(data_path=data_path, model_path=model_save_path)
test(data_path=data_path, model_path=model_save_path)