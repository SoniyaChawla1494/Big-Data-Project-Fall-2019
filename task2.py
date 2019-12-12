import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructType, StructField, IntegerType
import time
import json
import subprocess


sc = SparkContext()

spark = SparkSession.builder.appName("semantic_profiling").config('spark.driver.memory', '20g').config(
    'spark.executor.cores', 3).config('spark.executor.memory', '20g').config('spark.dynamicAllocation.enabled',True).config('spark.dynamicAllocation.maxExecutors', 25).config('spark.yarn.executor.memoryOverhead', '4096').getOrCreate()


def semantic_profiling(directory_path, process_start_time, datasets_to_run):

    def root_processing(df, column, num_dataset, function_list, column_name):

        def non_empty_cell_count(col):
            return F.count(col)

        def check_zipcode(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'^(1(0|1)\d{3})$') == True, col.cast(label_string)))

        def check_phone_number(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(
                r"^\s*[0-9]{10}\s*$|"r"^\s*([0-9]{3}|[0-9]{4}|[(][0-9]{3}[)]|[0-9](-*|\s*)[0-9]{3})(-*|\s*)[0-9]{3}(-*|\s*)[0-9]{4}\s*$") == True, col.cast(label_string)))

        def check_school_name(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'school|academy') == True, col.cast(label_string)))

        def check_street_name(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'avenue|lane|road|boulevard|drive|street|ave|dr|rd|blvd|ln|st') == True, col.cast(label_string)))

        def check_lat_lon(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r"^\s*[(]([0-9]|[1-8][0-9]|90)[.][0-9]+,\s-*([0-9]|[1-9][0-9]|1[0-7][0-9]|180)[.][0-9]+[)]\s*$") == True, col.cast(label_string)))

        def check_college_name(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'college|university') == True, col.cast(label_string)))

        def check_borough(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'^(manhattan|brooklyn|queens|bronx|staten island)$') == True, col.cast(label_string)))

        def check_park_playground(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'park') == True,col.cast(label_string)))

        def check_website(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(
                r'^(?:http|ftp)s?://' 
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' 
                r'localhost|' 
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' 
                r'(?::\d+)?' 
                r'(?:/?|[/?]\S+)$') == True,col.cast(label_string)))

        def check_business_name(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'corp|inc|llc|ltd') == True, col.cast(label_string)))

        def helper_subject(v):
            l = ['algebra', 'art', 'calculus', 'chemistry', 'cinema', 'composition', 'craft', 'drawing',
                 'global history', 'living environment', 'US GOVERNMENT & ECONOMICS',
                 'economics', 'economy', 'english', 'arts', 'geography', 'geometry', 'history', 'humanities',
                 'literature', 'math',
                 'mathematics', 'music', 'painting', 'physical development', 'physics', 'science', 'social science',
                 'social studies', 'biology']
            if v in l:
                return True
            else:
                return False

        def check_subject_school(col):
            return F.count(F.when(helper_subject(F.lower(col.cast(label_string))) == True, col.cast(label_string)))

        def check_neighbourhood(df_to_process):
            list_neighborhood = ["center", "village", "central", "bay", "west", "east", "north", "south",
                                 "upper", "lower", "side", "town", "cbd", "hill", "heights", "soho", "valley", "acres", "ridge", "harbor", "beach", "island", "club",
                                 "ferry", "mall", "oaks", "point", "hts", "neck", "yard", "basin","slope", "hook"]
            df_neighborhood_types = spark.createDataFrame(list_neighborhood, StringType()).select(
                F.collect_list("value")).withColumnRenamed(
                "collect_list(value)", "to_match")
            df_pre_neighborhood = df_neighborhood_types
            df_cross_join = df_to_process.crossJoin(df_pre_neighborhood)
            df_processed = df_cross_join.withColumn("size", F.size(F.array_intersect("token_filtered", "to_match")))
            df_street = df_processed.filter(df_processed.size != 0)
            df_left = df_processed.filter(df_processed.size == 0).drop("to_match").drop("size")
            return "neighborhood", df_left, df_street.select(F.sum("_c1"), F.lit('neighborhood').alias("sem_type"))

        label_string = 'string'

        def check_city_agency(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'system|tribunal|agency|library|council|board|commission|president|services|office|department|nyc|administration|authority') == True,col.cast(label_string)))

        def check_school_level(col):
            return F.count(F.when(F.lower(col.cast(label_string)).rlike(r'elementary|kindergarten|middle|nursery|primary|secondary|senior|high|transfer|k-2|k-3|k-8') == True,col.cast(label_string)))

        def check_address(col):
            return F.count(
                F.when(F.lower(col.cast(label_string)).rlike(r'\d+[ ](?:[A-Za-z0-9.-]+[ ]?)+(?:ave|dr|rd|blvd|ln|st|av)\.?') == True, col.cast(label_string)))

        def check_building_classification(col):
            return F.count(F.when(col.cast(label_string).rlike(r'^[A-Z][0-9A-Z][-][-\w]+$') == True, col.cast(label_string)))

        def helper_area_study(v):
            l = ['engineering', 'teaching', 'communications', 'animal ccience', 'science & math',
                 'law & government', 'architecture', 'business', 'culinary arts', 'performing arts',
                 'health profession', 'visual art & design', 'film/video', 'cosmetology',
                 'humanities & interdisciplinary', 'computer science & technology', 'project-based learning',
                 'hospitality, travel, & tourism', 'performing arts/visual art & design', 'environmental science',
                 'zoned']
            if v in l:
                return True
            else:
                return False

        def check_area_study(col):
            return F.count(F.when(helper_area_study(F.lower(col.cast(label_string))) == True, col.cast(label_string)))

        def check_middle_initial(col):
            return F.count(F.when(col.cast(label_string).rlike(r'^[A-Z][.]?$') == True, col.cast(label_string)))

        def helper_car_make(v):
            l = ['MERCEDES-BENZ S550', 'LEXUS ES 330', 'TOYOTA AVALON', 'HYUNDAI EQUUS', 'FORD FUSION',
                 'GMC YUKON DINALI', 'AUDI Q5', 'LINCOLN MKZ', 'HYUNDAI SONATA', 'MERCEDES-BENZ E350',
                 'FORD FREESTAR', 'MERCEDES-BENZ SPRINTER', 'CADILLAC ESCALADE', 'MERCURY MONTEREY', 'ACURA MDX',
                 'DODGE SPRINTER', 'CADILLAC XTS', 'MERCEDES-BENZ GL CLASS', 'GMC YUKON', 'LEXUS RX 350',
                 'NISSAN QUEST', 'HONDA ACCORD', 'CHEVROLET SUBURBAN', 'FORD EXPLORER', 'TOYOTA VENZA',
                 'GMC DENALI', 'NISSAN ROGUE', 'FORD', 'AUDI Q7', 'TOYOTA HIGHLANDER', 'LEXUS RX 400H',
                 'TOYOTA COROLLA', 'CHEVROLET CRUZE', 'CHEVROLET TRAILBLAZER', 'FORD WINDSTAR', 'BMW  7 SERIES',
                 'HONDA PILOT', 'CADILLAC ESCALADE ESV', 'VOLKSWAGEN ROUTAN', 'FORD FREESTYLE', 'LEXUS GX 460',
                 'CHRYSLER TOWN AND COUNTRY,TOYOTA HIGHLANDER', 'GMC YUKON XL', 'HONDA CR-V', 'LINCOLN NAVIGATOR',
                 'UNKNOWN', 'CHRYSLER 300', 'CHRYSLER TOWN AND COUNTRY', 'CHEVROLET MALIBU', 'DODGE CHARGER',
                 'FORD E350', 'CHEVROLET LUMINA', 'FORD TAURUS WAGON', 'JAGUAR XJ SERIES',
                 'CHRYSLER TOWN AND COUNTRY,LINCOLN TOWN CAR,', 'LEXUS LS 460', 'DODGE DURANGO', 'FORD FUSION SE',
                 'MERCEDES-BENZ E-CLASS', 'FORD ESCAPE', 'NISSAN NV200', 'BMW 5 SERIES', 'HONDA ODYSSEY',
                 'NISSAN PATHFINDER', 'LINCOLN MKT', 'TOYOTA RAV4', 'LINCOLN TOWN CAR', 'TOYOTA CAMRY',
                 'FORD EXPEDITION', 'BMW 7-SERIES', 'FORD CROWN VICTORIA', 'CHEVROLET EXPRESS,GMC YUKON XL',
                 'DODGE CARAVAN', 'TOYOTA PRIUS V', 'HONDA ODYSSEY,LINCOLN TOWN CAR,TOYOTA HIGHLANDER',
                 'NISSAN SENTRA', 'CHEVROLET EXPRESS LT 3500', 'MITSUBISHI MONTERO', 'GMC YUKON DENALI',
                 'JEEP GRAND CHEROKEE LAREDO', 'FORD FLEX', 'FORD ESCAPE,FORD FUSION,HONDA ODYSSEY', 'TOYOTA PRIUS',
                 'MERCEDES-BENZ S-CLASS', 'TOYOTA SIENNA', 'HYUNDAI VERACRUZ', 'MERCEDES-BENZ S350', 'NISSAN VERSA',
                 'FORD TAURUS', 'LEXUS ES 350', 'MERCEDES-BENZ R350', 'LINCOLN MKS', 'MERCURY GRAND MARQUIS',
                 'CHRYSLER PACIFICA', 'MERCEDES BENZ SPRINTER', 'FORD EXCURSION', 'FORD ESCAPE XLS 4WD',
                 'FORD C-MAX', 'CHEVROLET TAHOE', 'CHEVROLET UPLANDER', 'MERCEDES-BENZ', 'CADILLAC DTS',
                 'NISSAN ALTIMA', 'CHRYSLER SEBRING', 'FORD FIVE HUNDRED', 'CHEVROLET IMPALA', 'FORD EDGE',
                 'MAZDA MAZDA5']
            if v in l:
                return True
            else:
                return False

        def check_car_make(col):
            return F.count(F.when(helper_car_make(col.cast(label_string)) == True, col.cast(label_string)))

        def helper_location(v):
            l = ['STORE UNCLASSIFIED', 'PARK/PLAYGROUND', 'SOCIAL CLUB/POLICY', 'BUS STOP', 'HOMELESS SHELTER',
                 'AIRPORT TERMINAL',
                 'TRANSIT - NYC SUBWAY', 'PHOTO/COPY', 'CLOTHING/BOUTIQUE', 'CANDY STORE', 'MARINA/PIER',
                 'BUS TERMINAL', 'COMMERCIAL BUILDING',
                 'PARKING LOT/GARAGE (PRIVATE)', 'TRAMWAY', 'GAS STATION', 'LOAN COMPANY', 'ATM', 'PUBLIC SCHOOL',
                 'BAR/NIGHT CLUB', 'RESIDENCE-HOUSE', 'BOOK/CARD', 'BEAUTY & NAIL SALON', 'FOOD SUPERMARKET',
                 'BUS (NYC TRANSIT)', 'HIGHWAY/PARKWAY', 'SHOE', 'STORAGE FACILITY',
                 'CHAIN STORE', 'PRIVATE/PAROCHIAL SCHOOL', 'ABANDONED BUILDING', 'SYNAGOGUE', 'CEMETERY',
                 'FACTORY/WAREHOUSE', 'TUNNEL', 'HOTEL/MOTEL',
                 'SMALL MERCHANT', 'MAILBOX OUTSIDE', 'TAXI (YELLOW LICENSED)', 'RESIDENCE - APT. HOUSE',
                 'CONSTRUCTION SITE', 'MAILBOX INSIDE',
                 'VIDEO STORE', 'PARKING LOT/GARAGE (PUBLIC)', 'CHECK CASHING BUSINESS', 'VARIETY STORE', 'STREET',
                 'LIQUOR STORE', 'BUS (OTHER)',
                 'PUBLIC BUILDING', 'JEWELRY', 'RESTAURANT/DINER', 'OPEN AREAS (OPEN LOTS)', 'HOSPITAL',
                 'DAYCARE FACILITY', 'TELECOMM. STORE',
                 'MOSQUE', 'DEPARTMENT STORE', 'BANK', 'TAXI/LIVERY (UNLICENSED)', 'DRY CLEANER/LAUNDRY',
                 'TAXI (LIVERY LICENSED)', 'TRANSIT FACILITY (OTHER)', 'FAST FOOD''GROCERY/BODEGA', 'BRIDGE',
                 'DRUG STORE', 'FERRY/FERRY TERMINAL', 'OTHER HOUSE OF WORSHIP', 'RESIDENCE - PUBLIC HOUSING',
                 'OTHER', 'GYM/FITNESS FACILITY', 'DOCTOR/DENTIST OFFICE', 'CHURCH']
            if v in l:
                return True
            else:
                return False

        def check_location_type(col):
            return F.count(F.when(helper_location(col.cast(label_string)) == True, col.cast(label_string)))

        def helper_color(v):
            l = ['JADE', 'RED', 'RED APPLE PIN', 'WHITE', 'BLACK', '4 - COLOR LOGO', 'BLACK AND BLACK',
                 'GOLD APPLE PIN', 'GREEN', 'GRAY', 'HUNTER GREEN', 'ROYAL BLUE', 'SILVER', 'GOLD', 'MULTICOLOR',
                 'NAVY', 'BLUE PERIWINKLE', 'BROWN', 'BLACK AND GOLD', 'CLEAR', 'GREEN APPLE PIN', 'TURQUOISE',
                 'NAVY WITH ORANGE LOGO', 'BLUE', 'PINK', 'ORANGE', 'NAVY WITH WHITE LOGO']
            if v in l:
                return True
            else:
                return False

        def check_color(col):
            return F.count(F.when(helper_color(col.cast(label_string)) == True, col.cast(label_string)))

        function_list = [check_phone_number,check_school_name,
            check_lat_lon,check_college_name,check_borough,check_website,check_school_level,
            check_business_name,check_zipcode,check_subject_school,check_neighbourhood,check_city_agency,check_park_playground
            ,check_building_classification,check_area_study,check_middle_initial,check_car_make,check_location_type,
            check_color,check_street_name,non_empty_cell_count]

        main_labels = ['check_phone_number','check_school_name',
            'check_lat_lon','check_college_name','check_borough',
            'check_website','check_business_name','check_zipcode','check_subject_school','is_neighbourhood',
            'check_city_agency','check_park_playground','check_school_level','check_building_classification',
            'check_area_study','check_middle_initial','check_car_make',
            'check_location_type','check_color','check_street_name','number_of_non_empty_cells']

        def check_structure(df, col):
            def append_list():
                final_dict = []
                for fun in function_list:
                    final_dict.append(fun(F.col(col)).alias('{0}_{1}'.format(fun.__name__, col)))
                return final_dict

            list_iterator = iter(append_list())
            aggregated_ref = df.agg(*list_iterator).toJSON().first()
            json_list = json.loads(aggregated_ref)
            list_value = []
            for func in function_list:
                if '{0}_{1}'.format(func.__name__, col) in json_list.keys() and func.__name__ in main_labels:
                    if json_list['{0}_{1}'.format(func.__name__, col)] > 0:
                        list_value.append('{0}_{1}'.format(func.__name__, col))
            return list_value

        def get_s(df, c, l):

            col_names = df.columns

            def calculate():
                list_dict = []
                for func in function_list:
                    if '{0}_{1}'.format(func.__name__, c) in l:
                        list_dict.append(func(F.col(c)).alias('{0}_{1}'.format(func.__name__, c)))
                if len(list_dict) == 0:
                    list_dict.append(function_list[0](F.col(c)).alias('{0}_{1}'.format(function_list[0].__name__, c)))

                return list_dict

            iterator_list = iter(calculate())

            return df.agg(*iterator_list).toJSON().first()

        def get_l(s, column_name):

            json_list = json.loads(s)
            list_val = []
            for func in function_list:
                if '{0}_{1}'.format(func.__name__,
                                    column) in json_list.keys() and func.__name__ in main_labels and func.__name__ != 'number_of_non_empty_cells':
                    if json_list['{0}_{1}'.format(func.__name__, column)] > 0:
                        curr_dict = {}
                        curr_dict['semantic_type'] = func.__name__[func.__name__.index('_') + 1:]
                        curr_dict['count'] = json_list['{0}_{1}'.format(func.__name__, column)]
                        list_val.append(curr_dict)

            col_list = set(column_name.lower().split('_'))

            if {'last', 'name'} <= col_list or {'first',
                                              'name'} <= col_list:
                curr_dict = dict()
                curr_dict['semantic_type'] = 'person_name'
                curr_dict['count'] = json_list['{0}_{1}'.format('number_of_non_empty_cells', column)]
                list_val.append(curr_dict)

            if {'vehicle', 'body', 'type'} <= col_list:
                curr_dict = dict()
                curr_dict['semantic_type'] = 'vehicle_body_type'
                curr_dict['count'] = json_list['{0}_{1}'.format('number_of_non_empty_cells', column)]
                list_val.append(curr_dict)

            if {'vehicle', 'type',
                'code'} <= col_list:
                curr_dict = dict()
                curr_dict['semantic_type'] = 'vehicle_type_code'
                curr_dict['count'] = json_list['{0}_{1}'.format('number_of_non_empty_cells', column)]
                list_val.append(curr_dict)

            if len(list_val) == 0:
                curr_dict = dict()
                curr_dict['semantic_type'] = 'other'
                curr_dict['label'] = column_name
                curr_dict['count'] = json_list['{0}_{1}'.format('number_of_non_empty_cells', column)]
                list_val.append(curr_dict)

            return list_val

        if column:
            if num_dataset == 0:
                return check_structure(df, column)
            else:
                s = get_s(df, column, function_list)
                l = get_l(s, column_name)
                return l
        else:
            return []

    cmd = 'hdfs dfs -ls {}'.format(directory_path)
    files = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split('\n')
    files = list(files)
    files = files[1:]
    print('files success')

    with open('cluster1.txt', 'r') as f:
        x = f.readlines()

    l = x[0].strip('[').strip('\n').strip(']').split(',')
    column_list = []
    for z in l:
        t = z[z.index('\''):]
        column_list.append(t.strip('\''))

    i = 1
    for filename in column_list:
        if i <= datasets_to_run:

            f, c, _, _ = filename.split('.')
            t = {}
            df = spark.read.format('csv').option("delimiter", "\t").option("header", "false").option("inferschema","true").csv(str(directory_path + '/' + filename))

            if df.count() > 1000:
                df2 = df.sample(False, 0.1, seed=0).limit(100)
            else:
                df2 = df.sample(False, 0.9, seed=0).limit(100)
            k = root_processing(df2, "_c0", 0, [], c)

            t['column_name'] = c
            t['semantic_types'] = root_processing(df, "_c0", 1, k, c)

            df.unpersist()
            with open('task2.json', 'a') as fp:
                json.dump(t, fp)

            print("Completed dataset " + f + ' ' + str(
                time.time() - process_start_time))
            print("Completed {0} of {1}".format(i, datasets_to_run))
            process_start_time = time.time()
            i = i + 1
        else:
            break

    return 0


path = str(sys.argv[1])
n = int(sys.argv[2])

start_time = time.time()
semantic_profiling(path, start_time, n)
print("--- %s seconds ---" % (time.time() - start_time))

sc.stop()  # stop spark
