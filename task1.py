from pyspark.sql import SparkSession
from collections import Counter
import pyspark.sql.functions as F
import operator
import json
import subprocess
import sys
import time

spark = SparkSession.builder.appName("general_profiling_task1").config('spark.driver.memory', '12g').config('spark.resultant_dfecutor.cores', 4).config('spark.resultant_dfecutor.memory', '12g').getOrCreate()
def general_profiling_task1(path_to_folder, process_start_time, dataset_count):

    def column_labels_list():
        root_tags = ['count_non_empty_cells', 'count_empty_cells', 'count_distint_cells', 'get_frequent_value_count']
        int_tags = ['count_int_values', 'count_int_max_values', 'count_int_min_values', 'calculate_int_mean','calculate_int_stddev']
        real_tags = ['count_real_longest_values', 'count_real_max_values', 'count_real_min_values','calculate_real_mean', 'calculate_real_stddev']
        date_tags = ['count_date_values', 'count_date_max_values', 'count_date_min_values']
        str_tags = ['count_string_values', 'count_string_shortest_values', 'string_longest_values','calculate_string_avg']
        return root_tags,int_tags,real_tags,date_tags,str_tags

    def statistics_root(df):
        label_int = 'integer'
        label_double = 'double'
        label_string = 'string'
        date_format = 'MM/dd/yyyy'
        max_value = 500000

        def preprocess_column(column,type):
            return column.cast(type)

        def count_non_empty_cells(col):
            return F.count(col)

        def count_empty_cells(col):
            return F.count(F.when(F.isnull(col), col))

        def count_distint_cells(col):
            return F.approxCountDistinct(col)

        def compute_frequents(data):
            if len(data) < 0:
                return None
            else:
                count_values = Counter(data).most_common(5)
                frequents = [w[0] for w in count_values]
                return frequents
            
        def get_frequent_value_count(col):
            return compute_frequents(F.collect_list(col))

        def count_int_values(col):
            return F.when(F.count(col.cast(label_int)) == 0, None).otherwise(F.count(col.cast(label_int)))

        def count_int_max_values(col):
            return F.max(col.cast(label_int))

        def count_real_max_values(col):
            return F.max(col.cast(label_double))

        def count_date_max_values(col):
            return F.max(F.to_date(col.cast(label_string)))

        def count_int_min_values(col):
            return F.min(col.cast(label_int))

        def count_real_min_values(col):
            return F.min(col.cast(label_double))

        def count_date_min_values(col):
            return F.min(F.to_date(col.cast(label_string), date_format))

        def calculate_int_mean(col):
            return F.mean(col.cast(label_int))

        def calculate_real_mean(col):
            return F.mean(col.cast(label_double))

        def calculate_int_stddev(col):
            return F.stddev(col.cast(label_int))

        def calculate_real_stddev(col):
            return F.stddev(col.cast(label_double))

        def count_real_longest(col):
            return F.when(F.count(col.cast(label_double)) == 0, None).otherwise(F.count(col.cast(label_double)))

        def count_date_values(col):
            return F.when(F.count(F.to_date(col.cast(label_string), date_format)) == 0, None).otherwise(F.count(F.to_date(col.cast(label_string), date_format)))

        def count_string_values(col):
            return F.count(col.cast(label_string))

        def calculate_shortest_string(low, max_str):
            if len(low) < 0:
                return None
            else:
                if isinstance(low[0], str):
                    set_str = list(set(low))
                    set_str.sort(key=lambda s: len(s))
                    return set_str[:5]
                else:
                    return None

        def count_string_shortest_values(col):
            return calculate_shortest_string(F.collect_list(col), F.min(F.length(col)))

        def string_longest_values(col):
            return calculate_shortest_string(F.collect_list(col), F.max(F.length(col)))

        def calculate_string_avg(col):
            return F.avg(F.length(col))

        def calculate_df(df):

            function_list = [count_non_empty_cells, count_empty_cells, count_distint_cells, get_frequent_value_count,
                             count_int_values, count_int_max_values, count_int_min_values,
                             calculate_int_mean, calculate_int_stddev, count_real_longest, count_real_max_values,
                             count_real_min_values, calculate_real_mean, calculate_real_stddev, count_date_values,
                             count_date_max_values, count_date_min_values, count_string_values,
                             count_string_shortest_values, string_longest_values, calculate_string_avg]

            columns_df = df.columns_df

            df_schema = {}
            for curr_type in df.dtypes:
                df_schema[curr_type[0]] = curr_type[1]

            def apply_functions():
                resultant_list_functions = []
                for cols in columns_df:
                    f_format = f.__name__
                    format = '{0}_{1}'.format(f_format, cols)
                    if ('.' in cols):
                        continue
                    else:
                        for f in function_list:
                            if f_format in function_list:
                                resultant_list_functions.append(f(F.col(cols)).alias(format))
                            elif df_schema[cols] == label_string and f_format.split('_')[0] == label_string:
                                resultant_list_functions.append(f(F.col(cols)).alias(format))
                return resultant_list_functions

            resultant_df = iter(apply_functions())

            def create_dataframes():
                df_level1 = spark.createDataFrame([(['Root#'],)], ['Z'])
                df_level2 = spark.createDataFrame([(['Z'],)], ['Z'])
                df_level3 = spark.createDataFrame([(['Z'],)], ['Z'])
                return df_level1,df_level2,df_level3

            def collect_df():
                final_root_df = root_df.collect()
                final_df_level1 = df_level_1.collect()
                final_df_level2 = df_level_2.collect()
                return final_root_df,final_df_level1,final_df_level2

            freq_values = dict()
            longest_values = dict()
            shortest_values = dict()
            length_tag = "len"
            if df.count() > max_value:
                root_df ,df_level_1,df_level_2 = create_dataframes()

                for cols in df.columns_df:
                    if ('.' in cols):
                        continue
                    else:
                        root_df = root_df.union(df.groupBy(F.col(cols)).count().sort(F.desc("count")).limit(5).select(F.collect_list(F.col(cols))))
                        if df_schema[cols] == label_string:
                            df_level_2 = df_level_2.union(df.withColumn(length_tag, F.length(F.col(cols))).sort(F.desc(length_tag)).select(F.col(cols),length_tag).distinct().limit(5).select(F.collect_list(F.col(cols))))
                            df_level_1 = df_level_1.union(df.withColumn(length_tag, F.length(F.col(cols))).sort(F.asc(length_tag)).select(F.col(cols), length_tag).distinct().limit(5).select(F.collect_list(F.col(cols))))

                final_root_df ,final_df_level1,final_df_level2 = collect_df()

                count_1,count_2 = 1
                for cols in df.columns_df:
                    freq_values['{0}_{1}'.format('get_frequent_value_count', cols)] = final_root_df[count_1][0]
                    if df_schema[cols] == label_string:
                        longest_values['{0}_{1}'.format('longest_values', cols)] = final_df_level2[count_2][0]
                        shortest_values['{0}_{1}'.format('shortest_values', cols)] = final_df_level1[count_2][0]
                        count_2 = count_2 + 1
                    count_1 = count_1 + 1

            return df.agg(*resultant_df).toJSON().first(), freq_values, longest_values, shortest_values

        def format_edit(dict_name,cols):
            r = json.loads(string_s)
            dict_name[file_ptr.__name__[file_ptr.__name__.in_resultant_df('_') + 1:]] = r['{0}_{1}'.format(file_ptr.__name__, cols)]

        def formatting_column_names(col_name,cols):
            return '{0}_{1}'.format('col_name', cols)


        def get_l(s, freq_values, longest_values, shortest_values):
            function_list = [count_non_empty_cells, count_empty_cells, count_distint_cells, get_frequent_value_count,
                             count_int_values, count_int_max_values, count_int_min_values,
                             calculate_int_mean, calculate_int_stddev, count_real_longest, count_real_max_values,
                             count_real_min_values, calculate_real_mean, calculate_real_stddev, count_date_values,
                             count_date_max_values, count_date_min_values, count_string_values,
                             count_string_shortest_values, string_longest_values, calculate_string_avg]

            root_tag ,int_tag ,real_tag ,date_tag,string_tag = column_labels_list()

            df_schema = {}
            for sub_list in df.dtypes:
                df_schema[sub_list[0]] = sub_list[1]

            string_list = []
            json_load = json.loads(s)
            for cols in df.columns_df:
                main_dict = {}
                int_dictionary = dict()
                real_dictionary = dict()
                date_dictionary = dict()
                string_dictionary = dict()
                main_dict['column_name'] = cols
                int_dictionary['type'] = 'INTEGER (LONG)'
                real_dictionary['type'] = 'REAL'
                date_dictionary['type'] = 'DATE/TIME'
                string_dictionary['type'] = 'TEXT'
                for functions in function_list:
                    f_format = functions.__name__
                    pre_specified_format = '{0}_{1}'.format(f_format, cols)
                    if pre_specified_format in json_load.keys() and f_format in root_tag:
                        main_dict[f_format] = json_load[pre_specified_format]
                    elif pre_specified_format in json_load.keys() and f_format in int_tag:
                        format_edit(int_dictionary,cols)
                    elif pre_specified_format in json_load.keys() and f_format in real_tag:
                        format_edit(real_dictionary,cols)
                    elif pre_specified_format in json_load.keys() and f_format in date_tag:
                        format_edit(date_dictionary,cols)
                    elif pre_specified_format in json_load.keys() and f_format in string_tag:
                        format_edit(string_dictionary,cols)

                if bool(freq_values):
                    main_dict['get_frequent_value_count'] = freq_values[formatting_column_names('get_frequent_value_count',cols)]
                if bool(longest_values) and df_schema[cols] == 'string' and '{0}_{1}'.format('longest_values',cols) in longest_values.keys():
                    string_dictionary['longest_values'] = longest_values[formatting_column_names('longest_values,cols')]
                if bool(shortest_values) and df_schema[cols] == 'string' and '{0}_{1}'.format('shortest_values',cols) in shortest_values.keys():
                    string_dictionary['shortest_values'] = shortest_values[formatting_column_names('shortest_values',cols)]

                sub_list = []
                if 'count_int_values_{0}'.format(cols) in json_load.keys():
                    sub_list.append(int_dictionary)
                if 'count_real_longest_values_{0}'.format(cols) in json_load.keys():
                    sub_list.append(real_dictionary)
                if 'count_date_values_{0}'.format(cols) in json_load.keys():
                    sub_list.append(date_dictionary)
                if df_schema[cols] == label_string:
                    sub_list.append(string_dictionary)
                main_dict['data_types'] = sub_list
                string_list.append(main_dict)

            return string_list

        string_s, calculate_shortest_strings, file_sub, file_4 = calculate_df(df)
        string_l = get_l(string_s, calculate_shortest_strings, file_sub, file_4)

        return string_l

    cmd = 'hdfs dfs -ls {}'.format(path_to_folder)
    files = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split('\n')
    files = list(files)
    files = files[1:]
    data_order = {}
    for file_ptr in files:
        token = file_ptr.split(' ')
        data_order[token[-1]] = int(token[-4])
    data_order_x = sorted(data_order.items(),key=operator.itemgetter(1))
    print('files success')


    dataframe = spark.read.format('csv').option("delimiter", "\t").option("header", "false").option("inferdf_schema","true").csv(str(path_to_folder + '/' + 'datasets.tsv'))
    dataframe.createOrReplaceTempView("dataframe")
    if dataframe:
        print("dataset loaded successfully!")
    i = 1
    datasets_to_skip = [1144,1597, 1688,1689, 1697, 1717, 1767, 1772, 1787, 1788, 1789,1813,1817,1818,1825,1863]
    for file_index in data_order_x:
        if i <= num_datasets_to_compute:
            filename = file_index[0]
            find_slash = filename.rfind('/')
            if filename.endswith(".gz"):
                if i in datasets_to_skip:
                    i = i + 1
                    continue
                file_ptr = filename[find_slash:]
                file_ptr = file_ptr[1:]
                final_dictionary = dict()
                df = spark.read.format('csv').option("delimiter", "\t").option("header", "true").option("inferdf_schema", "true").csv(str(filename))
                dataset_name = spark.sql('select col1 as name from dataset where col0 = "{0}"'.format(str(file_ptr[:file_ptr.in_resultant_df(".")]))).toPandas()["name"][0]
                print("Dataset", str(file_ptr), "at in_resultant_df ", str(i), ", name: ", str(json.dumps(dataset_name)))
                print("No of rows: ", str(df.count()))
                print("No of columns_df: ", str(len(df.columns_df)))

                final_dictionary['dataset_name'] = dataset_name

                final_dictionary['columns_df'] = statistics_root(df)
                with open('task1.json', 'a') as fp:
                    json.dump(final_dictionary, fp)
                i = i + 1
        else:
            break

    return 0


path_to_folder = str(sys.argv[1])
num_datasets_to_compute = int(sys.argv[2])

process_start_time = time.time()
general_profiling_task1(path_to_folder, process_start_time, num_datasets_to_compute)
print("--- %s seconds ---" % (time.time() - process_start_time))
spark.stop()
