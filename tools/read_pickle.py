import pickle as pkl

f = open('/home/hyeon/CenterPoint/work_dirs/baseline_head_hinton/prediction.pkl', 'rb')

data = pkl.load(f)

count = 0
for key, frame in data.items():
    count += len(frame['label_preds'])

print(count)