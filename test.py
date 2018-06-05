with open("detectron_result_train_filter.txt", 'w') as outfile, open("detectron_result.train.txt", 'r') as infile:
    for line in infile:
        print(len(line))
        if len(line) > 2:
            outfile.write(line)
