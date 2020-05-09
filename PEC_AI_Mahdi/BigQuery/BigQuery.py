# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:05:26 2019

@author: Shin-Ting Wu, Hsin-Ting Chou
"""
import pandas as pd  
import numpy as np  
from google.cloud import bigquery
client = bigquery.Client(project="pec-ai-240309")

def Listing_datasets():
    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()
    datasets = list(client.list_datasets())
    project = client.project
    if datasets:
        print("Datasets in project {}:".format(project))
        for dataset in datasets:  # API request(s)
            print("\t{}".format(dataset.dataset_id))
    else:
        print("{} project does not contain any datasets.".format(project))

def Getting_a_dataset():
    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()
    # TODO(developer): Set dataset_id to the ID of the dataset to fetch.
    # dataset_id = 'your-project.your_dataset'
    dataset = client.get_dataset("pec-ai-240309.babynames")
    full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
    friendly_name = dataset.friendly_name
    print("Got dataset '{}' with friendly_name '{}'.".format(full_dataset_id, friendly_name))
    # View dataset properties
    print("Description: {}".format(dataset.description))
    print("Labels:")
    labels = dataset.labels
    if labels:
        for label, value in labels.items():
            print("\t{}: {}".format(label, value))
    else:
        print("\tDataset has no labels defined.")
        # View tables in dataset
        print("Tables:")
        tables = list(client.list_tables(dataset))  # API request(s)
        if tables:
            for table in tables:
                print("\t{}".format(table.table_id))
        else:
            print("\tThis dataset does not contain any tables.")
            
def Listing_tables():
    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()
    # TODO(developer): Set dataset_id to the ID of the dataset that contains
    #                  the tables you are listing.
    # dataset_id = 'your-project.your_dataset'
    dataset_id = "pec-ai-240309.babynames"
    tables = client.list_tables(dataset_id)
    print("Tables contained in '{}':".format(dataset_id))
    for table in tables:
        print("{}.{}.{}".format(table.project, table.dataset_id, table.table_id))
    
def Getting_a_table():
    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()
    # TODO(developer): Set table_id to the ID of the model to fetch.
    # table_id = 'your-project.your_dataset.your_table'
    table_id = "pec-ai-240309.babynames.names_2017"
    table = client.get_table(table_id)
    print("Got table '{}.{}.{}'.".format(table.project, table.dataset_id, table.table_id))
    # View table properties
    print("Table schema: {}".format(table.schema))
    print("Table description: {}".format(table.description))
    print("Table has {} rows".format(table.num_rows))
    
    dataset_ref = client.dataset("babynames")
    table_ref = dataset_ref.table("names_2017")
    table = client.get_table(table_ref)  # API call
    
    # Load all rows from a table
#    rows = client.list_rows(table)
#    assert len(list(rows)) == table.num_rows
    
    # Load the first 10 rows
#    rows = client.list_rows(table, max_results=10)
#    assert len(list(rows)) == 10
    
    # Specify selected fields to limit the results to certain columns
#    fields = table.schema[:2]  # first two columns
#    rows = client.list_rows(table, selected_fields=fields, max_results=10)
#    assert len(rows.schema) == 2
#    assert len(list(rows)) == 10
    
    # Use the start index to load an arbitrary portion of the table
    rows = client.list_rows(table, start_index=0, max_results=100)
    
    # Print row data in tabular format
    format_string = "{!s:<16} " * len(rows.schema)
    field_names = [field.name for field in rows.schema]
    print(format_string.format(*field_names))  # prints column headers
    for row in rows:
        print(format_string.format(*row))  # prints row data
        
def Creating_an_empty_table():
    schema = [
            bigquery.SchemaField("year", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("gender", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("count", "INTEGER", mode="NULLABLE"),
            ]

    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()
    # TODO(developer): Set table_id to the ID of the table to create
    # table_id = "your-project.your_dataset.your_table_name"
    
    table_id = "pec-ai-240309.babynames.BigqueryTOP100NameFrom2017"
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)  # API request
    print("Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id))
    
def Load_table_from_csv_file():
    # client = bigquery.Client()
    # filename = '/path/to/file.csv'
    # dataset_id = 'my_dataset'
    # table_id = 'my_table'
    
    filename = 'D:\\SpyderProjects\\PEC_AI\\BigqueryTOP100NameFrom2017.csv'
    dataset_id = "babynames"
    table_id = "BigqueryTOP100NameFrom2017"
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True
    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(
                source_file,
                table_ref,
                location="asia-east1",  # Must match the destination dataset location.
            job_config=job_config,
        )  # API request
    job.result()  # Waits for table load to complete.
    print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))
    
def Deleting_a_table():
    # TODO(developer): Construct a BigQuery client object.
    # client = bigquery.Client()

    # TODO(developer): Set table_id to the ID of the table to fetch.
    # table_id = 'your-project.your_dataset.your_table'

    # If the table does not exist, delete_table raises
    # google.api_core.exceptions.NotFound unless not_found_ok is True
    table_id = "pec-ai-240309.babynames.BigqueryTOP100NameFrom2017"
    client.delete_table(table_id, not_found_ok=True)
    print("Deleted table '{}'.".format(table_id))

def TOP100(File,Year,Gender):
    Nfile = pd.read_csv(File,header=None)
    Nfile.columns = ['Name', 'Gender', 'No']
    F = Nfile[(Nfile['Gender']==Gender)] 
    LenF=len(F)

    df=pd.DataFrame(F.sort_values('No', ascending= False))    
    df['Year']=[Year for i in range(0, LenF)]

    df = df.reindex(columns=['Year', 'Name', 'Gender', 'No'])
    df = df[0:100]
    return df

def TOP100_to_csv():
    df1=TOP100("C:\\Users\\user\\Desktop\\names\\yob2017.txt",'2017','F')
    df2=TOP100("C:\\Users\\user\\Desktop\\names\\yob2017.txt",'2017','M')
    df3=TOP100("C:\\Users\\user\\Desktop\\names\\yob2016.txt",'2016','F')
    df4=TOP100("C:\\Users\\user\\Desktop\\names\\yob2016.txt",'2016','M')
    df5=TOP100("C:\\Users\\user\\Desktop\\names\\yob2015.txt",'2015','F')
    df6=TOP100("C:\\Users\\user\\Desktop\\names\\yob2015.txt",'2015','M')
    df7=TOP100("C:\\Users\\user\\Desktop\\names\\yob2014.txt",'2014','F')
    df8=TOP100("C:\\Users\\user\\Desktop\\names\\yob2014.txt",'2014','M')
    df9=TOP100("C:\\Users\\user\\Desktop\\names\\yob2013.txt",'2013','F')
    df10=TOP100("C:\\Users\\user\\Desktop\\names\\yob2013.txt",'2013','M')
    df11=TOP100("C:\\Users\\user\\Desktop\\names\\yob2012.txt",'2012','F')
    df12=TOP100("C:\\Users\\user\\Desktop\\names\\yob2012.txt",'2012','M')
    df13=TOP100("C:\\Users\\user\\Desktop\\names\\yob2011.txt",'2011','F')
    df14=TOP100("C:\\Users\\user\\Desktop\\names\\yob2011.txt",'2011','M')
    df15=TOP100("C:\\Users\\user\\Desktop\\names\\yob2010.txt",'2010','F')
    df16=TOP100("C:\\Users\\user\\Desktop\\names\\yob2010.txt",'2010','M')
    df17=TOP100("C:\\Users\\user\\Desktop\\names\\yob2009.txt",'2009','F')
    df18=TOP100("C:\\Users\\user\\Desktop\\names\\yob2009.txt",'2009','M')
    df19=TOP100("C:\\Users\\user\\Desktop\\names\\yob2008.txt",'2008','F')
    df20=TOP100("C:\\Users\\user\\Desktop\\names\\yob2008.txt",'2008','M')
    DF =pd.concat([df1, df2, df3, df4, df5,
                   df6, df7, df8, df9, df10,
                   df11, df12, df13, df14, df15,
                   df16, df17, df18, df19, df20]) 
    print(DF)
    DF.to_csv('TOP100NameFrom2008to2017.csv', index=False) 
    
def Query1():
    query_job = client.query("""
                             SELECT
                             CONCAT(
                                     'https://stackoverflow.com/questions/',
                                     CAST(id as STRING)) as url,
                                     view_count
                             FROM `bigquery-public-data.stackoverflow.posts_questions`
                             WHERE tags like '%google-bigquery%'
                             ORDER BY view_count DESC
                             LIMIT 10""")
    results = query_job.result()  # Waits for job to complete.
    print(results)

def Query2():
    query_job = client.query("""
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2017`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2017`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2016`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2016`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2015`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2015`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2014`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2014`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2013`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2013`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2012`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2012`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2011`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2011`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2010`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2010`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2009`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2009`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2008`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100)
                             UNION ALL
                             (SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2008`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100)
                             """)

    results = query_job.result()  # Waits for job to complete.
    print(results)
    D = np.array([['Name', 'Gender', 'No']])
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
            #print(Year, row.name, row.gender, row.count)
    df=pd.DataFrame(D[1:], columns=D[0])
    year2017=['2017' for i in range(0,200)]
    year2016=['2016' for i in range(0,200)]
    year2015=['2015' for i in range(0,200)]
    year2014=['2014' for i in range(0,200)]
    year2013=['2013' for i in range(0,200)]
    year2012=['2012' for i in range(0,200)]
    year2011=['2011' for i in range(0,200)]
    year2010=['2010' for i in range(0,200)]
    year2009=['2009' for i in range(0,200)]
    year2008=['2008' for i in range(0,200)]

    year=np.concatenate((year2017, year2016, year2015, year2014, year2013,
                         year2012, year2011, year2010, year2009, year2008), axis=0)

    df['Year']=year
    df = df.reindex(columns=['Year', 'Name', 'Gender', 'No'])
    df.to_csv('BigqueryTOP100NameFrom2017.csv', index=False) 

def Query3():
    D = np.array([['Name', 'Gender', 'No']])
#2017F-------------------------------------------------------------------------
    query_job2017F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2017`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2017F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2017M-------------------------------------------------------------------------
    query_job2017M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2017`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2017M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2016F-------------------------------------------------------------------------        
    query_job2016F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2016`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2016F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2016M-------------------------------------------------------------------------        
    query_job2016M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2016`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2016M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2015F-------------------------------------------------------------------------    
    query_job2015F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2015`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2015F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2015M-------------------------------------------------------------------------            
    query_job2015M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2015`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2015M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2014F-------------------------------------------------------------------------    
    query_job2014F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2014`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2014F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2014M-------------------------------------------------------------------------            
    query_job2014M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2014`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2014M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2013F-------------------------------------------------------------------------    
    query_job2013F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2013`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2013F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2013M-------------------------------------------------------------------------            
    query_job2013M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2013`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2013M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2012F-------------------------------------------------------------------------            
    query_job2012F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2012`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2012F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2012M-------------------------------------------------------------------------            
    query_job2012M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2012`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2012M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)   
#2011F-------------------------------------------------------------------------    
    query_job2011F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2011`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2011F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2011M-------------------------------------------------------------------------            
    query_job2011M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2011`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2011M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2010F-------------------------------------------------------------------------    
    query_job2010F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2010`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2010F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2010M-------------------------------------------------------------------------            
    query_job2010M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2010`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2010M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2009F-------------------------------------------------------------------------    
    query_job2009F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2009`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2009F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2009M-------------------------------------------------------------------------            
    query_job2009M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2009`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2009M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2008F-------------------------------------------------------------------------    
    query_job2008F = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2008`
                             WHERE gender = 'F'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2008F.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
#2008M-------------------------------------------------------------------------            
    query_job2008M = client.query("""
                             SELECT name,gender,count
                             FROM `pec-ai-240309.babynames.names_2008`
                             WHERE gender = 'M'
                             ORDER BY count DESC
                             LIMIT 100
                             """)
    results = query_job2008M.result()  # Waits for job to complete.
    for row in results:
            D = np.append(D, [[row.name, row.gender, row.count]], axis=0)
    
    
    
    year2017=['2017' for i in range(0,200)]
    year2016=['2016' for i in range(0,200)]
    year2015=['2015' for i in range(0,200)]
    year2014=['2014' for i in range(0,200)]
    year2013=['2013' for i in range(0,200)]
    year2012=['2012' for i in range(0,200)]
    year2011=['2011' for i in range(0,200)]
    year2010=['2010' for i in range(0,200)]
    year2009=['2009' for i in range(0,200)]
    year2008=['2008' for i in range(0,200)]


    year=np.concatenate((year2017, year2016, year2015, year2014, year2013,
                         year2012, year2011, year2010, year2009, year2008), axis=0)
    
    df=pd.DataFrame(D[1:],columns=D[0])
    df['Year']=year
    df = df.reindex(columns=['Year', 'Name', 'Gender', 'No'])
#    print(df)
#    print(type(df))
    df.to_csv('BigqueryTOP100NameFrom2017.csv', index=False) 
#    df.to_gbq(destination_table = "babynames.BigqueryTOP100NameFrom2017", project_id = "pec-ai-240309")
    
#    # TODO(developer): Uncomment the lines below and replace with your values.
#    # from google.cloud import bigquery
#    # client = bigquery.Client()
#    # project = 'my_project'
#    # dataset_id = 'my_dataset'  # replace with your dataset ID
#    # table_id = 'my_table'  # replace with your table ID
#    # dataset_ref = client.dataset(dataset_id=dataset_id, project=project)
#    # dataset = client.get_dataset(dataset_ref)
#    # table_ref = dataset.table(table_id)

    project = "pec-ai-240309"
    dataset_id = "babynames"
    table_id = "BigqueryTOP100NameFrom2017"
    
    dataset_ref = client.dataset(dataset_id=dataset_id, project=project)
    dataset = client.get_dataset(dataset_ref)
    table_ref = dataset.table(table_id)
    
    client.load_table_from_dataframe(dataframe = df, destination = table_ref, project = project)
        
def main():
#    TOP100_to_csv()
#------------------------------------------------------------------------------
###    Query1()
###    Query2()
###    Query3()
#------------------------------------------------------------------------------
#    Listing_datasets()
#    Getting_a_dataset()
#    Listing_tables()
#    Getting_a_table()
#    Creating_an_empty_table()
#    Load_table_from_csv_file()
#    Deleting_a_table()
    return 1
main()