from curses import pair_content
from flask import Blueprint, jsonify
from zlib import Z_BEST_COMPRESSION
from flask import Flask, send_file, request
from flask_cors import CORS
from subprocess import PIPE, Popen
from partition.discrete.standard import standard as discrete_standard
from partition.continuous.standard import continuous_standard
from partition.continuous.agglomerative import continuous_agglomerative
from partition.continuous.agglomerative import AgglomerativeConfig
from io import BytesIO
import glob
import os
import random
import time
import tarfile
import shutil
import re 
import functools
import logging
import sys


app = Flask(__name__)
CORS(app)


class Instances(object):
    index = ""
    contentLineFile = []
    filename = ""

class StandardParams(object):
    def __init__(self, pf, pp, min_b, max_b) -> None:
        self.partition_factor = 10.0**int(pf)
        self.partition_percen = float(pp)
        self.min_bound = float(min_b)
        self.max_bound = float(max_b)

class AgglomerativeClusteringParams(object):
    def __init__(self, cluster_size, volumen_size, number_of_clusters, distance_method) -> None:
        self.cluster_size = float(cluster_size)
        self.volumen_size = float(volumen_size)
        self.number_of_clusters = number_of_clusters
        self.distance_method = distance_method

class Params(object):
    treelayout = False
    files = []
    names = []
    colors = []
    hash_file = ""
    def __init__(self, bmin, best, nruns, partition_value, nodesize, arrowsize, treelayout, files, names, colors, hash_file, typeproblem, strategy_partition, agglomerative_clustering, standard_configuration):
        self.bmin = bmin
        self.best = best
        self.nruns = nruns
        self.partition_value = partition_value
        self.nodesize = nodesize
        self.arrowsize = arrowsize
        self.treelayout = treelayout
        self.files = files
        self.names = names
        self.colors = colors
        self.hash_file = hash_file
        self.typeproblem = typeproblem
        self.strategy_partition = strategy_partition
        self.agglomerative_clustering = agglomerative_clustering
        self.standard_configuration = standard_configuration

def get_params() -> Params:
    bmin = request.form.get('bmin', "1") 
    best = request.form.get('best', "")
    nruns = request.form.get('nruns', "")
    hash_file = request.form.get('hash_file', "")
    partition_value = request.form.get('zvalue', "0.0")
    nodesize = request.form.get('nodesize', "1")
    arrowsize = request.form.get('arrowsize', "0.15")
    typeproblem = request.form.get('typeproblem', "")
    strategy_partition = request.form.get('strategy_partition', "standard")
    partition_factor = request.form.get('standard_configuration_partition_factor', 0)
    min_bound = request.form.get('standard_configuration_min_bound', 0.0)
    max_bound = request.form.get('standard_configuration_max_bound', 0.0)
    cluster_size = request.form.get('agglomerative_configuration_cluster_size', 50)
    volumen_size = request.form.get('agglomerative_configuration_volumen_size', 50)
    number_of_clusters = request.form.get('agglomerative_configuration_number_of_cluster', -1)
    distance_method = request.form.get('agglomerative_configuration_distance_method', "euclidean")

    print(number_of_clusters)
    if int(number_of_clusters) == -1:
        number_of_clusters = None
    else:
        number_of_clusters = int(number_of_clusters)
    
    print(bmin)
    print(typeproblem)
    print(strategy_partition)
    print(number_of_clusters)
    print(distance_method)
    print(partition_factor)
    print("volume: ", volumen_size)
    print("cluster_size: ", cluster_size)
    _files = request.files.getlist('file')
    _names = request.form.getlist('name[]')
    _colors = request.form.getlist('color[]')
    treelayout = True if request.form.get('treelayout', False) == "true" else False

    names = []
    colors = []
    files = []

    for n, c, f in sorted(zip(_names, _colors, _files), reverse=True):
        names.append(n)
        colors.append(c)
        files.append(f)
    
    agglomerative_clustering = AgglomerativeClusteringParams(cluster_size, volumen_size, number_of_clusters, distance_method)
    standard_configuration = StandardParams(partition_factor, partition_value, min_bound, max_bound)
    params =  Params(bmin, best, nruns, partition_value, nodesize, arrowsize, treelayout, files, names, colors, hash_file, typeproblem, strategy_partition, agglomerative_clustering, standard_configuration)

    if params.bmin != "":
        params.bmin = int(params.bmin)
    
    if params.partition_value != "":
        params.partition_value = float(params.partition_value)

    if params.nruns != "":
        if params.best != "":
            params.best = int(params.best)
        else: 
            params.best = 12
        params.nruns = int(params.nruns)
    return params
    
    
def change_old_format(a):
    a = [re.sub("(\s+)?[ ,\t](\s+)?", ",", a[i].strip()) for i in range(0, len(a))]
    new_content = [','.join(a[i].split(',')[:3]) + ',' + ','.join(a[i+1].split(',')[1:3]) for i in range(0, len(a)-1) 
    if len(a[i-1].split(',')) <= 3 and a[i].split(',')[0] == a[i+1].split(',')[0]]

    return new_content or a

def is_int(str):
    try: 
        int(str)
    except ValueError: 
        return False
    return True

def is_discrete(f):
    info = f[2].split(',')
    return is_int(info[2])


def writing_file_discrete(filename, contentLineFile, partition_value, strategy_partition, cfiles):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    start = time.time()

    file = discrete_standard(contentLineFile, partition_value, cfiles)

    end = time.time()
    print("Time: {}".format(end - start))
    if file[0] != "Run,Fitness1,Solution1,Fitness2,Solution2":
        file =  ["Run,Fitness1,Solution1,Fitness2,Solution2"] + file[:]
    with open(filename, "w") as f:
        for line in file:
            f.write("{}\n".format(line))

def writing_file_continuous(content_files, params, cfiles):
    if params.strategy_partition == "standard":
        results = continuous_standard(params, cfiles)
        for algo, results in results.items():
            for i, file in enumerate(results):
                if file[0] != "Run,Fitness1,Solution1,Fitness2,Solution2":
                    _file =  ["Run,Fitness1,Solution1,Fitness2,Solution2"] + file[:]
                else:
                    _file = file
                filename = "temp/{}/{}.csv".format(params.hash_file, algo)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    for line in _file:
                        f.write("{}\n".format(line))
                        
    elif params.strategy_partition == "agglomerative":
        results, min_clusters = continuous_agglomerative(params, cfiles)

        for algo, results_clustering in results.items():

            for i, file in enumerate(results_clustering.clustering):
                if file[0] != "Run,Fitness1,Solution1,Fitness2,Solution2":
                    _file =  ["Run,Fitness1,Solution1,Fitness2,Solution2"] + file[:]
                else:
                    _file = file
                filename = "temp/{}-{}/{}.csv".format(params.hash_file, results_clustering.number_of_clusters[i], algo)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    for line in _file:
                        f.write("{}\n".format(line))
                        
        params.hash_file = f"{params.hash_file}-{min_clusters}"

def process_partition(params):

    content_files = []
    for _ in range(len(params.files)):
        content_files.append(Instances())

    for index, f in enumerate(params.files):
        if ''.join(f.filename.split('.')[1:]).find("tar") > -1:
            with tarfile.open(name=None, fileobj=BytesIO(f.read())) as f_compress:
                for entry in f_compress:  
                    file_compress = f_compress.extractfile(entry)
                    content_files[index].contentLineFile = change_old_format(list(filter(None, file_compress.read().decode().split('\n'))))
        else:
            content_files[index].contentLineFile = change_old_format(list(filter(None, f.read().decode().split('\n'))))
        filename = "temp/{}/{}.csv".format(params.hash_file, params.names[index])
        if params.strategy_partition == "agglomerative" or params.typeproblem == "continuous":
            content_files[index].contentLineFile = [ params.names[index] + ',' + e for e in content_files[index].contentLineFile]
        content_files[index].filename = filename
        content_files[index].index = index

    all_solutions = functools.reduce(lambda i, j: i + j, [e.contentLineFile for e in content_files])
    print(params.typeproblem, "----", params.strategy_partition)

    if params.typeproblem == "discrete" and params.strategy_partition == 'standard':
        for file in content_files:
            writing_file_discrete(file.filename, file.contentLineFile, params.partition_value, params.strategy_partition, all_solutions)
    elif params.typeproblem == "discrete" and params.strategy_partition == 'agglomerative':
        writing_file_continuous(content_files, params, all_solutions)
    elif params.typeproblem == "continuous":
        writing_file_continuous(content_files, params, all_solutions)

    return params.hash_file

def generate_from_files(params : Params):
    print("--> Rscript create.R {} {} {} {}".format(params.hash_file, params.bmin, params.best, params.nruns))
    with Popen("Rscript create.R {} {} {} {}".format(params.hash_file, params.bmin, params.best, params.nruns), stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0]
        print("OK: {}".format(output))

    print("--> Rscript merge.R {}".format("{}-stn".format(params.hash_file)))
    with Popen("Rscript merge.R {}".format("{}-stn".format(params.hash_file)), stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0]
        print("OK: {}".format(output))

    print("--> Rscript plot-merged.R {} {} {} {}".format("{}-stn-merged.RData".format(params.hash_file), params.nodesize, params.arrowsize, " ".join(params.colors)))
    path = ""
    if params.treelayout:
        with Popen("Rscript plot-merged-tree.R {} {} {}".format("{}-stn-merged.RData".format(params.hash_file), params.nodesize, " ".join(["{}{}{}".format("\"", c, "\"") for c in params.colors])), stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0]
            print("OK: {}".format(output))
            path = "temp/{}-stn-merged-plot-tree.pdf".format(params.hash_file)
    else:
        with Popen("Rscript plot-merged.R {} {} {} {}".format("{}-stn-merged.RData".format(params.hash_file), params.nodesize, params.arrowsize, " ".join(["{}{}{}".format("\"", c, "\"") for c in params.colors])), stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0]
            print("OK: {}".format(output))
            path = "temp/{}-stn-merged-plot.pdf".format(params.hash_file)

    with Popen("Rscript metrics-merged.R {}".format("{}-stn-merged.RData".format(params.hash_file)), stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0]
        print("OK: {}".format(output))  
        shutil.rmtree("temp/" + params.hash_file)
        shutil.rmtree("temp/{}-stn".format(params.hash_file))

    return send_file(path, mimetype='application/pdf', as_attachment=True)

def generate_from_file(params : Params):
    print("--> Rscript create.R {} {} {} {}".format(params.hash_file, params.bmin, params.best, params.nruns))
    with Popen("Rscript create.R {} {} {} {}".format(params.hash_file, params.bmin, params.best, params.nruns), stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0]
        print("OK: {}".format(output))
    path = ""
    if params.treelayout:
        print("--> Rscript plot-alg-tree.R {} {}".format("{}-stn".format(params.hash_file), params.nodesize))
        with Popen("Rscript plot-alg-tree.R {} {}".format("{}-stn".format(params.hash_file), params.nodesize), stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0]
            print("OK: {}".format(output))
            path = "temp/{}-stn-plot-tree/{}_stn.pdf".format(params.hash_file, params.names[0])
    else:
        
        print("--> Rscript plot-alg.R {} {}".format("{}-stn".format(params.hash_file), params.nodesize))
        with Popen("Rscript plot-alg.R {} {}".format("{}-stn".format(params.hash_file), params.nodesize), stdout=PIPE, stderr=None, shell=True) as process:
            output = process.communicate()[0]
            print("OK: {}".format(output))
            path = "temp/{}-stn-plot/{}_stn.pdf".format(params.hash_file, params.names[0])

    with Popen("Rscript metrics-alg.R {}".format("{}-stn".format(params.hash_file)), stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0]
        print("OK: {}".format(output))  
    return send_file(path, mimetype='application/pdf', as_attachment=True)


@app.route("/agglomerative-info", methods=['POST'])
def get_agglomerative_info():
    hash_file = request.form.get('hash_file', "")
    print(hash_file)
    clusters = [int(e.split('-')[-1]) for e in list(filter(re.compile(f"{hash_file}-[0-9]+$").match, os.listdir("temp")))]

    
    if clusters:
        return jsonify(
            limit_init = min(clusters),
            limit_end = max(clusters)
        )
    else:
        number_one_cluster = [int(e.split('-')[1]) for e in list(filter(re.compile(f"{hash_file}-[0-9]+").match, os.listdir("temp")))][0]
        return jsonify(
            limit_init = number_one_cluster,
            limit_end = number_one_cluster
        )

@app.route("/stn-metrics", methods=['POST'])
def get_metrics():
    hash_file = "temp/" + request.form.get('hash_file', "")
    print(hash_file)
    return send_file(hash_file, mimetype='text/csv', as_attachment=True)

@app.route("/stn", methods=['POST'])
def generate():
    params : Params = get_params()
    if params.agglomerative_clustering.number_of_clusters:
        params.hash_file = f"{params.hash_file}-{params.agglomerative_clustering.number_of_clusters}" 
    else:
        process_partition(params)
    
    if len(params.files) > 1:
        return generate_from_files(params)
    else:
        return generate_from_file(params)

errors = Blueprint('errors', __name__)

@errors.app_errorhandler(Exception)
def handle_unexpected_error(error):
    status_code = 500
    success = False
    response = {
        'success': success,
        'error': {
            'type': 'UnexpectedException',
            'message': 'An unexpected error has occurred.'
        }
    }

    return jsonify(response), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=True) 
