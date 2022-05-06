from collections import OrderedDict
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import random
import os
from os.path import exists
from tempfile import TemporaryFile
import networkx as nx


def load_data(filename, bucket):
    if(exists(filename)):
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                curr_val = float(row['transition']) - float(row['dwell'])
                time = (curr_val) // 300
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


def mds_embeddings(x):
    seed = np.random.RandomState(seed=3)
    mds = MDS(n_components=2, metric=False, max_iter=300, eps=1e-12,
              dissimilarity="precomputed", random_state=seed)
    pos = mds.fit_transform(x)
    return pos


def get_cmap(n):
    return plt.cm.get_cmap('hsv', n)


def display_b_results(results, buildings):
    n = len(buildings)
    colors = get_cmap(n)

    fig, ax = plt.subplots()

    for i in range(n):
        x = results[i][0]
        y = results[i][1]
        ax.scatter(x, y, color=colors(i), label=buildings[i])
        ax.annotate(buildings[i], (x, y))

    plt.show()


def dissimilarity_util(arr_connected):
    abs_path = '/Users/kaushalrai/Desktop/UWM/Independent Study/WiFi Map-selected/transitions/transitions/'
    matrix = [[0 for i in range(len(arr_connected))]
              for j in range(len(arr_connected))]
    for ind in range(len(arr_connected)):
        print(arr_connected[ind])
        for ind1 in range(ind + 1, len(arr_connected)):
            bucket = {}
            file_to_read_1 = abs_path + \
                arr_connected[ind] + '/' + arr_connected[ind1] + ".csv"
            file_to_read_2 = abs_path + \
                arr_connected[ind1] + '/' + arr_connected[ind] + ".csv"

            if (exists(file_to_read_1) or exists(file_to_read_2)):
                load_data(file_to_read_1, bucket)
                load_data(file_to_read_2, bucket)
                max_count = 0
                res_val = 0

                for i in bucket:
                    if bucket[i][0] > max_count:
                        max_count = bucket[i][0]
                        res_val = bucket[i][1]
                matrix[ind][ind1] = res_val
                matrix[ind1][ind] = res_val
            else:
                matrix[ind][ind1] = 0
                matrix[ind1][ind] = 0
    return matrix


def create_default_dissimilarity():
    no_cord = ['walnuthc', 'smi', 'kronshage', 'tripp', '432nm',
               'adams', 'usqr', 'moore', '1410jd', 'chartrhc', 'king', '.idea']
    directories = os.walk(abs_path)
    arr_new = []
    arr_temp = []
    for x in directories:
        array_temp = x[1]
        break

    for i in range(len(array_temp)):
        if array_temp[i][0] != '.' and array_temp[i] not in no_cord:
            arr_new.append(array_temp[i])

    matrix = dissimilarity_util(arr_new)

    np.savetxt("arr_overall.txt", arr_new, fmt='%s')
    np.savetxt("dissimilarity_matrix_247.txt", matrix, fmt='%f')
    print("saved")


def create_dissimilarity_matrix(out, arr_new):
    arr_connected = []
    for idx, val in enumerate(out):
        print(val)
        arr_connected.append(arr_new[val])
    matrix = dissimilarity_util(arr_connected)
    # arr_connected = ['witte', 'lakeshore', 'nicholas','enghall', 'unsth', 'memlib' ,'cssc', 'memun']
    np.savetxt("arr_connected_final.txt", arr_connected, fmt='%s')
    np.savetxt("dissimilarity_matrix_connected_final.txt", matrix, fmt='%f')
    true_distance_matrix = create_true_distance_matrix(arr_connected)
    correlation_plot(dissimilarity_matrix, true_distance_matrix)
    x = mds_embeddings(matrix)
    return arr_connected, x


def calculate_err(arr_connected_final, locations):
    bldng1, bldng2 = "enghall", "memlib"
    dissimilarity = np.loadtxt(
        "/Users/abhinayreddy/Downloads/dissimilarity_matrix_connected_final.txt")
    x = mds_embeddings(dissimilarity)
    index = []
    for ind in range(len(arr_connected)):
        if arr_connected[ind] == "enghall" or arr_connected[ind] == "memlib":
            index.append(ind)
    fst, snd = index[0], index[1]
    x1, y1 = x[fst][0], x[fst][1]
    x_1, y_1 = locations[fst][1][0], locations[fst][1][1]

    x2, y2 = x[snd][0], x[snd][1]
    x_2, y_2 = locations[snd][1][0], locations[snd][1][1]

    S_x = (x_1 - x_2) / (x1 - x2)
    a = x_1 - (S_x*x1)

    S_y = (y_1 - y_2) / (y1 - y2)
    b = y_1 - (S_y*y1)
    dist = 0
    lat_list = []
    long_list = []
    display_list = []

    for ind in range(len(locations)):
        proj_x = (S_x * x[ind][0] + a)
        proj_y = (S_y * x[ind][1] + b)
        lat_list.append(proj_x)
        long_list.append(proj_y)
        # label.append()
        display_list.append([proj_x, proj_y])
        dist += geodesic((proj_x, proj_y), locations[ind][1]).miles
        print(arr_connected[ind], dist)
    gmap = gmplot.GoogleMapPlotter(43.0688898, -89.4065277, 14)
    for each in range(len(lat_list)):
        gmap.text(lat_list[each], long_list[each], arr_connected[each])
        gmap.text(locations[each][1][0], locations[each][1]
                  [1], arr_connected[each], color="blue")

    gmap.scatter(lat_list, long_list,
                 color='#3B0B39', size=40, marker=False)
    gmap.draw('map_comp.html')

    display_b_results(display_list, arr_connected)
    avg_error = (dist/len(arr_connected))


def correlation_plot(dissimilarity_matrix, true_matrix):
    matrix = dissimilarity_matrix
    true_matrix = np.array(true_matrix)
    x = np.log10(matrix[np.triu_indices(135, k=1)])
    y = np.log10(true_matrix[np.triu_indices(135, k=1)])

    plt.scatter(x, y)
    m, b = np.polyfit(x, y, 1)
    plt.xlabel('Log 10 Observation Buckets')
    plt.ylabel('Log 10 Distance')
    plt.plot(x, m*x + b, color='black')
    plt.show()


def create_true_distance_matrix(arr_temp):
    filename = "/Users/abhinayreddy/Documents/cs799/transitions/true_locations.csv"
    for building in arr_temp:
        # print(building, "building")
        with open(filename, newline='') as csvfile:
            reader_n = csv.DictReader(csvfile)
            for row in reader_n:
                # print(row, "row")
                if building == row['building']:

                    if row['latitude'] != "" and row['longitude'] != "":
                        locations.append(
                            (row['building'], (float(row['latitude']), float(row['longitude']))))
                    else:
                        count += 1
                        no_coord.append(building)
                        locations.append((row['building'], (0, 0)))

    #locations = calculate_err(arr_temp, locations)
    n = (len(locations))
    calculate_err(arr_temp, locations)

    true_matrix = [[0.0 for i in range(n)] for j in range(n)]
    for source in range(n):
        for dest in range(source+1, n):
            # print(locations[source], locations[dest], "source n dest")
            dist = geodesic(locations[source], locations[dest]).miles
            # print(dist, locations[source][1], locations[dest][1], end="\n")
            true_matrix[source][dest] = dist
            true_matrix[dest][source] = dist

            # pairwise_dist[dest][source] = pairwise_dist[source][dest]
        # print(pairwise_dict,"dist")
    print(true_matrix, "mat")

    return true_matrix


if __name__ == "__main__":
    create_default_dissimilarity()
    a = np.loadtxt(
        "/Users/kaushalrai/Desktop/UWM/Independent Study/WiFi Map-selected/transitions/transitions/dissimilarity_matrix_247.txt")
    arr_new = np.loadtxt(
        "/Users/kaushalrai/Desktop/UWM/Independent Study/WiFi Map-selected/transitions/transitions/arr_overall.txt", dtype='str')
    G = nx.from_numpy_array(a)
    out = nx.algorithms.approximation.max_clique(G)
    print("max_clique formed")
    load_final_array, load_final_matrix = create_dissimilarity_matrix(
        out, arr_new)
    # load_final_matrix = np.loadtxt("/Users/kaushalrai/Desktop/UWM/Independent Study/WiFi Map-selected/transitions/transitions/dissimilarity_matrix_connected_135.txt")
    # load_final_array = np.loadtxt("/Users/kaushalrai/Desktop/UWM/Independent Study/WiFi Map-selected/transitions/transitions/arr_connected_135.txt", dtype='str')
    x = mds_embeddings(load_final_matrix)
    image_plot = display_b_results(x, load_final_array)
