import os
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

seeds = ['1','2','7','8','9']
# ratios1 = ['0.95', '0.96', '0.97', '0.98', '0.99']
# ratios2 = ['0.05', '0.04', '0.03', '0.02', '0.01']
ratios1 = ['0.9']
ratios2 = ['0.1']

# for ratio 1
for r in range(len(ratios1)):
    tensorboard_dir = os.path.join('./change_test/test_avg', 'test_' + ratios1[r] + '_opp_' + ratios1[r] + '_'  + ratios2[r])
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    d = defaultdict(float)
    for i in seeds:
        file_dir = './change_test/test_' + i + '/test_' + ratios1[r] + '_opp_' + ratios1[r] + '_'  + ratios2[r] + '_' + i
        file_name = os.path.join(file_dir, os.listdir(file_dir)[0])
        print(file_name)
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        mean_scores = ea.scalars.Items('test/mean_score')
        for m in mean_scores:
            if m.step < 200000:
                d[m.step] += m.value
    for i in d:
        writer.add_scalar("test/mean_score", d[i] / len(seeds), i)

# for ratio 2
for r in range(len(ratios1)):
    tensorboard_dir = os.path.join('./change_test/test_avg', 'test_' + ratios2[r] + '_opp_' + ratios1[r] + '_'  + ratios2[r])
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    d = defaultdict(float)
    for i in seeds:
        file_dir = './change_test/test_' + i + '/test_' + ratios2[r] + '_opp_' + ratios1[r] + '_'  + ratios2[r] + '_' + i
        file_name = os.path.join(file_dir, os.listdir(file_dir)[0])
        print(file_name)
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        mean_scores = ea.scalars.Items('test/mean_score')
        for m in mean_scores:
            if m.step < 200000:
                d[m.step] += m.value
    for i in d:
        writer.add_scalar("test/mean_score", d[i] / len(seeds), i)