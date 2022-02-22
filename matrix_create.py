from collections import OrderedDict
import csv


def load_data(filename, bucket):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #        print(row)
            #print(row['transition'], row['dwell'])
            curr_val = float(row['transition']) - float(row['dwell'])
            time = (curr_val) // 300
            print(time, "time")
            if time <= 12:
                if time in bucket:
                    val = ((bucket[time][1] * bucket[time][0]) +
                           curr_val) / (bucket[time][0] + 1)
                    bucket[time][0] += 1
                    bucket[time][1] = val
                else:
                    bucket[time] = [0, 0]
                    bucket[time][0] = 1
                    bucket[time][1] = curr_val


if __name__ == "__main__":
    arr = ['cssc', 'memun']

    matrix = [[0 for i in range(2)] for j in range(2)]

    for ind in range(len(arr)):
        #print(ind, "h1")
        for ind1 in range(ind + 1, len(arr)):
            # print(ind, ind1, "h")
            bucket = {}
            file_to_read_1 = arr[ind] + '/' + arr[ind1] + ".csv"
            file_to_read_2 = arr[ind1] + '/' + arr[ind] + ".csv"
            print(file_to_read_1, "F")
            load_data(file_to_read_1, bucket)
            print(bucket, "1st")
            load_data(file_to_read_2, bucket)
            cnt_values = bucket.values()
            print(cnt_values, "v")
            max_count = 0

            for i in bucket:
                if bucket[i][0] > max_count:
                    max_count = bucket[i][0]
                    res_val = bucket[i][1]

            matrix[ind][ind1] = res_val
            matrix[ind1][ind] = res_val
            print(matrix, "Final")
